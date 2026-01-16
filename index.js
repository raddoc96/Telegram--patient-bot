import { Telegraf } from 'telegraf';
import { message } from 'telegraf/filters';
import { GoogleGenerativeAI } from '@google/generative-ai';
import express from 'express';
import mongoose from 'mongoose';
import ffmpeg from 'fluent-ffmpeg';
import ffmpegInstaller from '@ffmpeg-installer/ffmpeg';
import axios from 'axios';
import fs from 'fs';
import os from 'os';
import { join } from 'path';

// --- CONFIGURATION ---
ffmpeg.setFfmpegPath(ffmpegInstaller.path);

const CONFIG = {
    // Split keys by comma
    API_KEYS: (process.env.GEMINI_API_KEYS || '').split(',').map(k => k.trim()).filter(k => k),
    TELEGRAM_TOKEN: process.env.TELEGRAM_BOT_TOKEN,
    GEMINI_MODEL: 'gemini-2.0-flash',
    MONGODB_URI: process.env.MONGODB_URI,
    MEDIA_TIMEOUT_MS: 300000, // 5 minutes to accumulate files
    CONTEXT_RETENTION_MS: 1800000, // 30 minutes memory for replies
    MAX_STORED_CONTEXTS: 20,
    SYSTEM_INSTRUCTION: `You are an expert medical AI assistant specializing in radiology. You have two modes of operation:

**MODE 1: CLINICAL PROFILE GENERATION**
When provided with medical files (images, PDFs, audio recordings, or video files) and/or text context, you extract and analyze all content to create a concise and comprehensive "Clinical Profile".

IMPORTANT INSTRUCTION - IF THE HANDWRITTEN TEXT IS NOT LEGIBLE, FEEL FREE TO USE CODE INTERPRETATION AND LOGIC IN THE CONTEXT OF OTHER TEXTS TO DECIPHER THE ILLEGIBLE TEXT

FOR AUDIO FILES: Transcribe the audio content carefully and extract all relevant medical information mentioned.

FOR VIDEO FILES: Analyze the video content, transcribe any audio, and extract all visible medical information including any text, scans, or documents shown.

FOR TEXT MESSAGES: These may contain additional clinical context, patient history, or notes that should be incorporated into the Clinical Profile.

YOUR RESPONSE MUST BE BASED SOLELY ON THE PROVIDED CONTENT (files AND text).

Follow these strict instructions for Clinical Profile generation:

Analyze All Content: Meticulously examine all provided files - images, PDFs, audio recordings, and video files, as well as any accompanying text messages. This may include prior medical scan reports (like USG, CT, MRI), clinical notes, voice memos, video recordings, or other relevant documents.

Extract Key Information: From the content, identify and extract all pertinent information, such as:
- Scan types (e.g., USG, CT Brain).
- Dates of scans or documents.
- Key findings, measurements, or impressions from reports.
- Relevant clinical history mentioned in notes, audio, video, or text messages.

Synthesize into a Clinical Profile:
- Combine all extracted information into a single, cohesive paragraph. This represents a 100% recreation of the relevant clinical details from the provided content.
- If there are repeated or vague findings across multiple documents, synthesize them into a single, concise statement.
- Frame sentences properly to be concise, but you MUST NOT omit any important clinical details. Prioritize completeness of clinical information over extreme brevity.
- You MUST strictly exclude any mention of the patient's name, age, or gender.
- If multiple dated scan reports are present, you MUST arrange their summaries chronologically in ascending order based on their dates.
- If a date is not available for a scan, refer to it as "Previous [Scan Type]...".

Formatting for Clinical Profile:
- The final output MUST be a single paragraph.
- This paragraph MUST start with "Clinical Profile:" and the entire content (including the prefix) must be wrapped in single asterisks. For example: "*Clinical Profile: Previous USG dated 01/01/2023 showed mild hepatomegaly. Patient also has a H/o hypertension as noted in the clinical sheet.*"

Do not output the raw transcribed text.
Do not output JSON or Markdown code blocks.
Return ONLY the single formatted paragraph described above.

**MODE 2: FOLLOW-UP INTERACTION**
When a user replies to a previously generated Clinical Profile, you should:

1. If the user ASKS A QUESTION (e.g., "What does this mean?", "Can you explain the findings?", "What is hepatomegaly?", "Is this serious?"):
   - Answer the question directly and helpfully based on the Clinical Profile and the original medical content
   - Provide clear, understandable explanations
   - If appropriate, explain medical terms in simple language
   - Be informative but remind that this is AI analysis, not medical advice

2. If the user PROVIDES ADDITIONAL CONTEXT or CORRECTIONS (e.g., "The patient also has diabetes", "There was another report showing..."):
   - Incorporate the new information into the Clinical Profile
   - Generate an UPDATED Clinical Profile following the same format rules as MODE 1

3. If the user sends ADDITIONAL FILES in the reply:
   - Analyze the new files along with the original context
   - Generate an UPDATED Clinical Profile that includes information from all files

IMPORTANT: Always identify whether the user is asking a question or providing additional information, and respond appropriately.`
};

// --- DATA STRUCTURES ---
const userBuffers = new Map(); // Stores pending files per chat
const bufferTimeouts = new Map(); // Auto-clear timers
const contextHistory = new Map(); // Stores past bot responses for context

// --- MONGODB SETUP ---
const contextSchema = new mongoose.Schema({
    chatId: String,
    messageId: String, // The bot's response ID
    originalMedia: Array, // Metadata about files sent
    responseText: String,
    timestamp: { type: Date, default: Date.now }
});
// Auto-delete after retention period
contextSchema.index({ timestamp: 1 }, { expireAfterSeconds: CONFIG.CONTEXT_RETENTION_MS / 1000 });
const ContextModel = mongoose.model('Context', contextSchema);

async function connectMongoDB() {
    if (!CONFIG.MONGODB_URI) return console.log('⚠️ No MongoDB URI - Contexts wont persist restarts');
    try {
        await mongoose.connect(CONFIG.MONGODB_URI);
        console.log('✅ MongoDB Connected');
    } catch (e) {
        console.error('❌ MongoDB Error:', e.message);
    }
}

// --- HELPER FUNCTIONS ---

function getBuffer(chatId) {
    if (!userBuffers.has(chatId)) userBuffers.set(chatId, []);
    return userBuffers.get(chatId);
}

function clearBuffer(chatId) {
    if (bufferTimeouts.has(chatId)) {
        clearTimeout(bufferTimeouts.get(chatId));
        bufferTimeouts.delete(chatId);
    }
    const items = userBuffers.get(chatId) || [];
    userBuffers.delete(chatId);
    return items;
}

function resetTimeout(chatId, ctx) {
    if (bufferTimeouts.has(chatId)) clearTimeout(bufferTimeouts.get(chatId));
    
    bufferTimeouts.set(chatId, setTimeout(async () => {
        const items = clearBuffer(chatId);
        if (items.length > 0) {
            await ctx.reply(`⏰ Timeout: Cleared ${items.length} pending files from buffer.`);
        }
    }, CONFIG.MEDIA_TIMEOUT_MS));
}

// --- SMART VIDEO PROCESSING (Oversample + Filter) ---
async function extractFramesFromVideo(videoBuffer, targetFps = 3) {
    return new Promise((resolve, reject) => {
        const tempId = Math.random().toString(36).substring(7);
        const tempDir = os.tmpdir();
        const inputPath = join(tempDir, `input_${tempId}.mp4`);
        const outputPattern = join(tempDir, `frame_${tempId}_%03d.jpg`);

        fs.writeFileSync(inputPath, videoBuffer);

        // INTELLIGENT FILTER: Grab 3x frames, pick sharpest
        const batchSize = 3;
        const inputFps = targetFps * batchSize;
        const videoFilter = `fps=${inputFps},thumbnail=${batchSize}`;

        console.log(`🎬 Smart Extract: Target ${targetFps}fps`);

        ffmpeg(inputPath)
            .outputOptions([
                `-vf ${videoFilter}`,
                '-vsync 0',
                '-q:v 2' // High quality JPG
            ])
            .output(outputPattern)
            .on('end', () => {
                try {
                    const files = fs.readdirSync(tempDir)
                        .filter(f => f.startsWith(`frame_${tempId}_`) && f.endsWith('.jpg'))
                        .sort();

                    const frames = files.map(file => {
                        const path = join(tempDir, file);
                        const buffer = fs.readFileSync(path);
                        fs.unlinkSync(path);
                        return buffer.toString('base64');
                    });

                    fs.unlinkSync(inputPath);
                    resolve(frames);
                } catch (err) {
                    reject(err);
                }
            })
            .on('error', (err) => {
                try { if (fs.existsSync(inputPath)) fs.unlinkSync(inputPath); } catch(e){}
                reject(err);
            })
            .run();
    });
}

// --- GEMINI PROCESSING ENGINE ---
async function processWithGemini(chatId, items, isFollowUp = false, previousContext = null, userQuery = null, targetFps = 3) {
    
    // 1. Pre-process Videos
    const processedItems = [];
    let counts = { images: 0, pdfs: 0, video: 0, audio: 0, text: 0 };

    for (const item of items) {
        if (item.type === 'video') {
            try {
                // Download video buffer
                const response = await axios.get(item.url, { responseType: 'arraybuffer' });
                const videoBuffer = Buffer.from(response.data);
                
                const frames = await extractFramesFromVideo(videoBuffer, targetFps);
                console.log(`📸 Extracted ${frames.length} frames`);
                
                frames.forEach(frame => {
                    processedItems.push({
                        type: 'image',
                        data: frame, // base64
                        mime: 'image/jpeg',
                        caption: item.caption ? `[Video Frame] ${item.caption}` : '[Video Frame]'
                    });
                    counts.images++;
                });
            } catch (e) {
                console.error('Video process error, falling back to raw video not supported in this mode');
            }
        } else if (item.type === 'image' || item.type === 'pdf' || item.type === 'audio') {
            try {
                // For images/PDFs, we need base64 for Gemini
                const response = await axios.get(item.url, { responseType: 'arraybuffer' });
                processedItems.push({
                    type: item.type,
                    data: Buffer.from(response.data).toString('base64'),
                    mime: item.mime,
                    caption: item.caption
                });
                if (item.type === 'image') counts.images++;
                if (item.type === 'pdf') counts.pdfs++;
                if (item.type === 'audio') counts.audio++;
            } catch (e) { console.error('Download error', e); }
        } else if (item.type === 'text') {
            processedItems.push(item);
            counts.text++;
        }
    }

    // 2. Construct Prompt
    const contentParts = [];
    const textNotes = [];

    for (const item of processedItems) {
        if (item.type === 'text') {
            textNotes.push(item.text);
        } else {
            contentParts.push({
                inlineData: { data: item.data, mimeType: item.mime }
            });
            if (item.caption) textNotes.push(`[Caption]: ${item.caption}`);
        }
    }

    let prompt = "";
    
    if (isFollowUp && previousContext) {
        const isQ = isQuestion(userQuery);
        if (isQ) {
            prompt = `User asks a QUESTION about the previous profile.
=== PREVIOUS PROFILE ===
${previousContext}
=== ORIGINAL CONTEXT ===
${textNotes.join('\n')}
=== USER QUESTION ===
${userQuery}

Answer the question directly based on the context.`;
        } else {
            prompt = `User provides UPDATE/CORRECTION.
=== PREVIOUS PROFILE ===
${previousContext}
=== NEW INFO ===
${userQuery}
${textNotes.join('\n')}

Generate UPDATED Clinical Profile.`;
        }
    } else {
        prompt = `Analyze these medical files.
=== NOTES ===
${textNotes.join('\n')}

Generate the Clinical Profile.`;
    }

    const requestParts = [prompt, ...contentParts];

    // 3. Call API (Rotation Logic)
    let responseText = null;
    const keys = CONFIG.API_KEYS;
    
    if (keys.length === 0) return "❌ No API Keys configured.";

    for (let i = 0; i < keys.length; i++) {
        try {
            const genAI = new GoogleGenerativeAI(keys[i]);
            const model = genAI.getGenerativeModel({ 
                model: CONFIG.GEMINI_MODEL, 
                systemInstruction: CONFIG.SYSTEM_INSTRUCTION 
            });
            const result = await model.generateContent(requestParts);
            responseText = result.response.text();
            break; 
        } catch (e) {
            console.log(`⚠️ Key ${i} failed: ${e.message}`);
        }
    }

    if (!responseText) throw new Error("All API keys failed.");
    return responseText;
}

function isQuestion(text) {
    if (!text) return false;
    const t = text.toLowerCase().trim();
    return t.endsWith('?') || 
           ['what', 'why', 'how', 'is', 'does', 'can', 'explain'].some(w => t.startsWith(w));
}

// --- TELEGRAM BOT LOGIC ---

if (!CONFIG.TELEGRAM_TOKEN) {
    console.error("❌ No TELEGRAM_BOT_TOKEN provided!");
    process.exit(1);
}

const bot = new Telegraf(CONFIG.TELEGRAM_TOKEN);

// 1. Start Command
bot.command('start', (ctx) => {
    ctx.reply(`🏥 *Medical Bot Ready*
    
1️⃣ Send Images, PDFs, Audio, or *Fast Videos*
2️⃣ Send command to process:
   • *.*  (Smart 3 FPS - Default)
   • *.2* (Smart 2 FPS)
   • *.1* (Smart 1 FPS)
   
↩️ Reply to my messages to ask questions!`, { parse_mode: 'Markdown' });
});

// 2. Clear Command
bot.command('clear', (ctx) => {
    const items = clearBuffer(ctx.chat.id);
    ctx.reply(`🗑️ Cleared ${items.length} items from buffer.`);
});

// 3. Handle Messages
bot.on(message('text'), async (ctx) => {
    const text = ctx.message.text.trim();
    const chatId = ctx.chat.id;

    // A. Check for Triggers
    if (['.', '.1', '.2'].includes(text)) {
        const items = clearBuffer(chatId);
        if (items.length === 0) return ctx.reply("⚠️ Buffer empty. Send files first.");

        let fps = 3;
        if (text === '.1') fps = 1;
        if (text === '.2') fps = 2;

        const loadingMsg = await ctx.reply(`⏳ Processing ${items.length} items (Smart ${fps} FPS)...`);
        
        try {
            const result = await processWithGemini(chatId, items, false, null, null, fps);
            
            // Send Long Message (Split if needed)
            const parts = result.match(/[\s\S]{1,4000}/g) || [];
            let lastMsg;
            for (const part of parts) {
                lastMsg = await ctx.reply(part, { parse_mode: 'Markdown' });
            }

            // Save Context
            if (lastMsg) {
                await ContextModel.create({
                    chatId: String(chatId),
                    messageId: String(lastMsg.message_id),
                    originalMedia: items.map(i => ({ type: i.type, mime: i.mime })),
                    responseText: result
                });
            }
            ctx.telegram.deleteMessage(chatId, loadingMsg.message_id).catch(()=>{});

        } catch (e) {
            ctx.reply(`❌ Error: ${e.message}`);
        }
        return;
    }

    // B. Check for Reply (Follow-up)
    if (ctx.message.reply_to_message) {
        const replyId = String(ctx.message.reply_to_message.message_id);
        const context = await ContextModel.findOne({ chatId: String(chatId), messageId: replyId });

        if (context) {
            const loadingMsg = await ctx.reply("🔄 Analyzing reply...");
            try {
                // If user attached files in reply (Telegram handles text+media separately, 
                // but usually text reply is just text). 
                // We assume just text reply for simplicity or check buffer? 
                // For simplicity, we just take the text as query.
                const newResponse = await processWithGemini(
                    chatId, 
                    [], // No new media in text-only reply
                    true, 
                    context.responseText, 
                    text
                );
                
                const sent = await ctx.reply(newResponse, { parse_mode: 'Markdown' });
                
                // Update context chain
                await ContextModel.create({
                    chatId: String(chatId),
                    messageId: String(sent.message_id),
                    originalMedia: context.originalMedia,
                    responseText: newResponse
                });
                
                ctx.telegram.deleteMessage(chatId, loadingMsg.message_id).catch(()=>{});

            } catch (e) {
                ctx.reply(`❌ Error: ${e.message}`);
            }
            return;
        }
    }

    // C. Buffer Text Notes
    getBuffer(chatId).push({ type: 'text', text: text });
    resetTimeout(chatId, ctx);
    ctx.reply(`📝 Text note added. (${getBuffer(chatId).length} items pending)`);
});

// 4. Handle Media
const handleMedia = async (ctx, type) => {
    const chatId = ctx.chat.id;
    let fileId, mime, caption;

    if (type === 'photo') {
        const photos = ctx.message.photo;
        fileId = photos[photos.length - 1].file_id; // Get best quality
        mime = 'image/jpeg';
        caption = ctx.message.caption;
    } else if (type === 'document') {
        fileId = ctx.message.document.file_id;
        mime = ctx.message.document.mime_type;
        caption = ctx.message.caption;
        if (!mime.includes('pdf')) return ctx.reply("⚠️ Only PDFs supported for documents.");
    } else if (type === 'video') {
        fileId = ctx.message.video.file_id;
        mime = ctx.message.video.mime_type;
        caption = ctx.message.caption;
    } else if (type === 'voice' || type === 'audio') {
        const obj = ctx.message.voice || ctx.message.audio;
        fileId = obj.file_id;
        mime = obj.mime_type || 'audio/ogg';
        caption = ctx.message.caption;
    }

    // Get Link
    try {
        const fileLink = await ctx.telegram.getFileLink(fileId);
        
        getBuffer(chatId).push({
            type: type === 'photo' ? 'image' : type,
            url: fileLink.href,
            mime: mime,
            caption: caption || ''
        });

        resetTimeout(chatId, ctx);
        ctx.reply(`📎 ${type.toUpperCase()} added. (${getBuffer(chatId).length} items pending)`);
    } catch (e) {
        ctx.reply("❌ Failed to get file link. Is it too big?");
    }
};

bot.on(message('photo'), ctx => handleMedia(ctx, 'photo'));
bot.on(message('document'), ctx => handleMedia(ctx, 'document'));
bot.on(message('video'), ctx => handleMedia(ctx, 'video'));
bot.on(message('voice'), ctx => handleMedia(ctx, 'voice'));
bot.on(message('audio'), ctx => handleMedia(ctx, 'audio'));

// --- STARTUP ---
(async () => {
    await connectMongoDB();
    
    // Web Server for Render Health Check
    const app = express();
    app.get('/', (req, res) => res.send('Telegram Medical Bot Running'));
    app.get('/health', (req, res) => res.json({status: 'ok'}));
    app.listen(process.env.PORT || 3000);

    // Launch Bot
    bot.launch(() => console.log('🚀 Telegram Bot Started'));
    
    // Graceful Stop
    process.once('SIGINT', () => bot.stop('SIGINT'));
    process.once('SIGTERM', () => bot.stop('SIGTERM'));
})();
