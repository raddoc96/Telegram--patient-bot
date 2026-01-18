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

// ==============================================================================
// 🟢 USER CONFIGURATION AREA
// ==============================================================================

// 1. ORIGINAL RADIOLOGY INSTRUCTION (For '.' commands)
const PRIMARY_SYSTEM_INSTRUCTION = `You are an expert medical AI assistant specializing in radiology. You have two modes of operation:

**MODE 1: CLINICAL PROFILE GENERATION**
When provided with medical files (images, PDFs, audio recordings, or video files) and/or text context, you extract and analyze all content to create a concise and comprehensive "Clinical Profile".

IMPORTANT INSTRUCTION - IF THE HANDWRITTEN TEXT IS NOT LEGIBLE, FEEL FREE TO USE CODE INTERPRETATION AND LOGIC IN THE CONTEXT OF OTHER TEXTS TO DECIPHER THE ILLEGIBLE TEXT

FOR AUDIO FILES: Transcribe the audio content carefully and extract all relevant medical information mentioned.
FOR VIDEO FILES: Analyze the video content, transcribe any audio, and extract all visible medical information including any text, scans, or documents shown.
FOR TEXT MESSAGES: These may contain additional clinical context, patient history, or notes that should be incorporated into the Clinical Profile.

YOUR RESPONSE MUST BE BASED SOLELY ON THE PROVIDED CONTENT (files AND text).

Follow these strict instructions for Clinical Profile generation:
Analyze All Content: Meticulously examine all provided files.
Extract Key Information: Scan types, Dates, Key findings, Clinical history.

Synthesize into a Clinical Profile:
- Combine all extracted information into a single, cohesive paragraph.
- You MUST strictly exclude any mention of the patient's name, age, or gender.
- If multiple dated scan reports are present, arrange chronologically.

Formatting for Clinical Profile:
- The final output MUST be a single paragraph.
- This paragraph MUST start with "Clinical Profile:" and the entire content (including the prefix) must be wrapped in single asterisks. For example: "*Clinical Profile: Previous USG dated 01/01/2023 showed mild hepatomegaly...*"

**MODE 2: FOLLOW-UP INTERACTION**
If user ASKS A QUESTION: Answer directly based on Clinical Profile.
If user PROVIDES CONTEXT: Generate UPDATED Clinical Profile.`;

// 2. NEW SECONDARY INSTRUCTION (For '..' commands)
// TODO: Fill this with your new system instruction
const SECONDARY_SYSTEM_INSTRUCTION = `You are an expert radiologist.When you receive a context, it is mostly about a patient and sometimes they might have been advised with any imaging modality. You analyse that info and then advise regarding that as an expert radiologist what to be seen in that specific imaging modality for that specific patient including various hypothetical imaging findings from common to less common for that patient condition in that specific imaging modality. suppose of you cant indentify thr specific imaging modality in thr given context, you yourself choose the appropriate imaging modality based on the specific conditions context`;

// 3. PROMPT TO TRIGGER SECONDARY BOT
// TODO: Fill this with the specific prompt you want to send along with the profile
const SECONDARY_TRIGGER_PROMPT = `Here is the Clinical Profile generated from the patient's reports. Please analyze this profile according to your system instructions and provide the final output.`;

// ==============================================================================


const CONFIG = {
    API_KEYS: (process.env.GEMINI_API_KEYS || '').split(',').map(k => k.trim()).filter(k => k),
    TELEGRAM_TOKEN: process.env.TELEGRAM_BOT_TOKEN,
    GEMINI_MODEL: 'gemini-3-flash-preview',
    MONGODB_URI: process.env.MONGODB_URI,
    MEDIA_TIMEOUT_MS: 300000, 
    CONTEXT_RETENTION_MS: 1800000, 
    MAX_STORED_CONTEXTS: 20
};

// --- DATA STRUCTURES ---
const userBuffers = new Map(); 
const bufferTimeouts = new Map(); 

// --- MONGODB SETUP ---
const contextSchema = new mongoose.Schema({
    chatId: String,
    messageId: String, 
    originalMedia: Array, 
    responseText: String, 
    mode: { type: String, enum: ['primary', 'secondary'], default: 'primary' }, 
    timestamp: { type: Date, default: Date.now }
});
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

// --- SAFE MESSAGE SENDER (FIXES ERROR 400) ---
async function sendSafeMessage(ctx, text, prefix = "") {
    const MAX_LENGTH = 4000; // Leave buffer for safety
    const fullText = prefix + text;
    const chunks = [];

    let remainingText = fullText;
    while (remainingText.length > 0) {
        if (remainingText.length <= MAX_LENGTH) {
            chunks.push(remainingText);
            break;
        }
        // Try to split at the nearest newline to avoid cutting words/tags in half
        let splitIndex = remainingText.lastIndexOf('\n', MAX_LENGTH);
        if (splitIndex === -1) splitIndex = MAX_LENGTH; // No newline found, hard split

        chunks.push(remainingText.substring(0, splitIndex));
        remainingText = remainingText.substring(splitIndex).trim(); // Remove leading whitespace
    }

    let lastSentMsg;
    for (const chunk of chunks) {
        try {
            // 1. Try sending with Markdown
            lastSentMsg = await ctx.reply(chunk, { parse_mode: 'Markdown' });
        } catch (e) {
            // 2. If Markdown fails (likely due to split tags), fallback to Plain Text
            // This prevents the "Can't find end of entity" error
            // console.log("Markdown failed, retrying plain text...");
            lastSentMsg = await ctx.reply(chunk);
        }
    }
    return lastSentMsg;
}

// --- SMART VIDEO PROCESSING ---
async function extractFramesFromVideo(videoBuffer, targetFps = 3) {
    return new Promise((resolve, reject) => {
        const tempId = Math.random().toString(36).substring(7);
        const tempDir = os.tmpdir();
        const inputPath = join(tempDir, `input_${tempId}.mp4`);
        const outputPattern = join(tempDir, `frame_${tempId}_%03d.jpg`);

        fs.writeFileSync(inputPath, videoBuffer);

        const batchSize = 3;
        const inputFps = targetFps * batchSize;
        const videoFilter = `fps=${inputFps},thumbnail=${batchSize}`;

        console.log(`🎬 Smart Extract: Target ${targetFps}fps`);

        ffmpeg(inputPath)
            .outputOptions([`-vf ${videoFilter}`, '-vsync 0', '-q:v 2'])
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

// --- GEMINI CORE ENGINE ---
async function generateGeminiResponse(contentParts, systemInstruction) {
    const keys = CONFIG.API_KEYS;
    if (keys.length === 0) throw new Error("No API Keys");

    for (let i = 0; i < keys.length; i++) {
        try {
            const genAI = new GoogleGenerativeAI(keys[i]);
            const model = genAI.getGenerativeModel({ 
                model: CONFIG.GEMINI_MODEL, 
                systemInstruction: systemInstruction 
            });
            const result = await model.generateContent(contentParts);
            return result.response.text();
        } catch (e) {
            console.log(`⚠️ Key ${i} failed: ${e.message}`);
        }
    }
    throw new Error("All API keys failed.");
}

// --- MAIN PROCESSOR ---
async function processRequest(chatId, items, mode, previousContext = null, userQuery = null, targetFps = 3) {
    
    // 1. Prepare Media
    const processedContent = [];
    const textNotes = [];
    
    if (items.length > 0) {
        for (const item of items) {
            if (item.type === 'video') {
                try {
                    const response = await axios.get(item.url, { responseType: 'arraybuffer' });
                    const videoBuffer = Buffer.from(response.data);
                    const frames = await extractFramesFromVideo(videoBuffer, targetFps);
                    frames.forEach(frame => {
                        processedContent.push({ inlineData: { data: frame, mimeType: 'image/jpeg' } });
                        if (item.caption) textNotes.push(`[Video Frame Caption]: ${item.caption}`);
                    });
                } catch (e) { console.error('Video error'); }
            } else if (['image', 'pdf', 'audio'].includes(item.type)) {
                try {
                    const response = await axios.get(item.url, { responseType: 'arraybuffer' });
                    processedContent.push({
                        inlineData: { 
                            data: Buffer.from(response.data).toString('base64'), 
                            mimeType: item.mime 
                        }
                    });
                    if (item.caption) textNotes.push(`[${item.type} caption]: ${item.caption}`);
                } catch (e) { console.error('Download error'); }
            } else if (item.type === 'text') {
                textNotes.push(item.text);
            }
        }
    }

    // --- LOGIC BRANCHING ---

    // A. FOLLOW-UP (Reply)
    if (previousContext) {
        let prompt = "";
        const instruction = previousContext.mode === 'secondary' 
            ? SECONDARY_SYSTEM_INSTRUCTION 
            : PRIMARY_SYSTEM_INSTRUCTION;

        if (isQuestion(userQuery)) {
            prompt = `User asks a QUESTION about the previous output.
=== PREVIOUS OUTPUT ===
${previousContext.responseText}
=== NEW CONTEXT ===
${textNotes.join('\n')}
=== USER QUESTION ===
${userQuery}
Answer the question directly based on the context.`;
        } else {
            prompt = `User provides UPDATE/CORRECTION.
=== PREVIOUS OUTPUT ===
${previousContext.responseText}
=== NEW INFO ===
${userQuery}
${textNotes.join('\n')}
Generate UPDATED output.`;
        }

        const request = [prompt, ...processedContent];
        const response = await generateGeminiResponse(request, instruction);
        return { response, mode: previousContext.mode };
    }

    // B. NEW REQUEST - PRIMARY (Trigger: .)
    if (mode === 'primary') {
        const prompt = `Analyze these medical files.
=== NOTES ===
${textNotes.join('\n')}
Generate the Clinical Profile.`;
        
        const request = [prompt, ...processedContent];
        const response = await generateGeminiResponse(request, PRIMARY_SYSTEM_INSTRUCTION);
        return { response, mode: 'primary' };
    }

    // C. NEW REQUEST - SECONDARY/CHAINED (Trigger: ..)
    if (mode === 'secondary') {
        // Step 1: Run Primary Logic (Internal)
        const promptPrimary = `Analyze these medical files.
=== NOTES ===
${textNotes.join('\n')}
Generate the Clinical Profile.`;
        
        const requestPrimary = [promptPrimary, ...processedContent];
        const primaryResponse = await generateGeminiResponse(requestPrimary, PRIMARY_SYSTEM_INSTRUCTION);
        
        // Step 2: Feed Primary Result to Secondary Bot
        const promptSecondary = `${SECONDARY_TRIGGER_PROMPT}

=== CLINICAL PROFILE ===
${primaryResponse}
=== END PROFILE ===`;

        const requestSecondary = [promptSecondary];
        const secondaryResponse = await generateGeminiResponse(requestSecondary, SECONDARY_SYSTEM_INSTRUCTION);
        
        // Return BOTH responses
        return { 
            response: secondaryResponse, 
            primaryResponse: primaryResponse, 
            mode: 'secondary' 
        };
    }
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

bot.command('start', (ctx) => {
    ctx.reply(`🏥 *Medical Bot Ready*
    
1️⃣ Send Files (Images, PDF, Video)
2️⃣ Send command:
   • *.*  (Standard Clinical Profile)
   • *..* (Secondary Analysis Chain)
   
   (Add numbers 1 or 2 for video speed, e.g., .2 or ..2)
   
↩️ Reply to messages to continue context!`, { parse_mode: 'Markdown' });
});

bot.command('clear', (ctx) => {
    const items = clearBuffer(ctx.chat.id);
    ctx.reply(`🗑️ Cleared ${items.length} items.`);
});

bot.on(message('text'), async (ctx) => {
    const text = ctx.message.text.trim();
    const chatId = ctx.chat.id;

    // 1. CHECK TRIGGERS
    const isPrimary = /^(\.|(\.[1-3]))$/.test(text);
    const isSecondary = /^(\.\.|(\.\.[1-3]))$/.test(text);

    if (isPrimary || isSecondary) {
        const items = clearBuffer(chatId);
        if (items.length === 0) return ctx.reply("⚠️ Buffer empty. Send files first.");

        const lastChar = text.slice(-1);
        let fps = isNaN(lastChar) ? 3 : parseInt(lastChar);
        if (text === '.' || text === '..') fps = 3;

        const mode = isSecondary ? 'secondary' : 'primary';
        const label = isSecondary ? 'CHAINED Analysis' : 'Clinical Profile';

        const loadingMsg = await ctx.reply(`⏳ Processing ${items.length} items (${label}, Smart ${fps} FPS)...`);
        
        try {
            const result = await processRequest(chatId, items, mode, null, null, fps);
            
            // IF SECONDARY MODE: Send the Primary Response (Clinical Profile) First!
            // We use sendSafeMessage to handle splitting and Markdown errors
            if (result.primaryResponse) {
                await sendSafeMessage(ctx, result.primaryResponse, "📝 *Clinical Profile (Step 1):*\n\n");
            }

            // Send Final Response (Primary or Secondary)
            const prefix = result.primaryResponse ? "🧠 *Secondary Analysis (Step 2):*\n\n" : "";
            const lastMsg = await sendSafeMessage(ctx, result.response, prefix);

            // Save Context
            if (lastMsg) {
                await ContextModel.create({
                    chatId: String(chatId),
                    messageId: String(lastMsg.message_id),
                    originalMedia: items.map(i => ({ type: i.type, mime: i.mime })),
                    responseText: result.response,
                    mode: result.mode 
                });
            }
            ctx.telegram.deleteMessage(chatId, loadingMsg.message_id).catch(()=>{});

        } catch (e) {
            ctx.reply(`❌ Error: ${e.message}`);
        }
        return;
    }

    // 2. CHECK REPLY
    if (ctx.message.reply_to_message) {
        const replyId = String(ctx.message.reply_to_message.message_id);
        const context = await ContextModel.findOne({ chatId: String(chatId), messageId: replyId });

        if (context) {
            const loadingMsg = await ctx.reply(`🔄 Analyzing reply (${context.mode} context)...`);
            try {
                const result = await processRequest(
                    chatId, [], context.mode, context, text
                );
                
                const lastMsg = await sendSafeMessage(ctx, result.response);
                
                await ContextModel.create({
                    chatId: String(chatId),
                    messageId: String(lastMsg.message_id),
                    originalMedia: context.originalMedia,
                    responseText: result.response,
                    mode: context.mode 
                });
                
                ctx.telegram.deleteMessage(chatId, loadingMsg.message_id).catch(()=>{});

            } catch (e) {
                ctx.reply(`❌ Error: ${e.message}`);
            }
            return;
        }
    }

    // 3. BUFFER TEXT
    getBuffer(chatId).push({ type: 'text', text: text });
    resetTimeout(chatId, ctx);
    ctx.reply(`📝 Text note added.`);
});

// MEDIA HANDLERS
const handleMedia = async (ctx, type) => {
    const chatId = ctx.chat.id;
    let fileId, mime, caption;

    if (type === 'photo') {
        const photos = ctx.message.photo;
        fileId = photos[photos.length - 1].file_id;
        mime = 'image/jpeg';
        caption = ctx.message.caption;
    } else if (type === 'document') {
        fileId = ctx.message.document.file_id;
        mime = ctx.message.document.mime_type;
        caption = ctx.message.caption;
        if (!mime.includes('pdf')) return ctx.reply("⚠️ Only PDFs supported.");
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

    try {
        const fileLink = await ctx.telegram.getFileLink(fileId);
        getBuffer(chatId).push({
            type: type === 'photo' ? 'image' : type,
            url: fileLink.href,
            mime: mime,
            caption: caption || ''
        });
        resetTimeout(chatId, ctx);
        ctx.reply(`📎 ${type.toUpperCase()} added.`);
    } catch (e) {
        ctx.reply("❌ Error getting file.");
    }
};

bot.on(message('photo'), ctx => handleMedia(ctx, 'photo'));
bot.on(message('document'), ctx => handleMedia(ctx, 'document'));
bot.on(message('video'), ctx => handleMedia(ctx, 'video'));
bot.on(message('voice'), ctx => handleMedia(ctx, 'voice'));
bot.on(message('audio'), ctx => handleMedia(ctx, 'audio'));

// STARTUP
(async () => {
    await connectMongoDB();
    const app = express();
    app.get('/', (req, res) => res.send('Telegram Medical Bot V3.1 Running'));
    app.get('/health', (req, res) => res.json({status: 'ok'}));
    app.listen(process.env.PORT || 3000);

    bot.launch(() => console.log('🚀 Telegram Bot Started'));
    process.once('SIGINT', () => bot.stop('SIGINT'));
    process.once('SIGTERM', () => bot.stop('SIGTERM'));
})();
