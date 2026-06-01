import { Telegraf } from 'telegraf';
import { message } from 'telegraf/filters';
import { GoogleGenerativeAI } from '@google/generative-ai';
import express from 'express';
import ffmpeg from 'fluent-ffmpeg';
import ffmpegInstaller from '@ffmpeg-installer/ffmpeg';
import axios from 'axios';
import fs from 'fs';
import os from 'os';
import { join } from 'path';

// Setup FFmpeg path automatically
ffmpeg.setFfmpegPath(ffmpegInstaller.path);

// Helper to parse multiple Gemini keys
const getApiKeys = () => {
  const keys = process.env.GEMINI_API_KEYS || process.env.GEMINI_API_KEY || '';
  return keys.split(',').map(k => k.trim()).filter(k => k.length > 0);
};

// ======================================================================
// 🟢 CONFIGURATION AREA
// ======================================================================

const PRIMARY_SYSTEM_INSTRUCTION = `You are an expert medical AI assistant specializing in radiology. 

**CLINICAL PROFILE GENERATION**
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

IMPORTANT ADDITIONAL OUTPUT:
After the Clinical Profile paragraph, you MUST output a second line (separated by a blank line) in EXACTLY this format:
<<JSON>>{"mrn":"<Registration Number/MRN or Not mentioned>","age":"<age or unknown>","sex":"<M/F/unknown>","study":"<imaging study indicated or Not mentioned>","brief":"<very concise reason for scan using abbreviations like H/o, C/o, K/c/o, etc., mentioning duration of symptoms>"}<<JSON>>

Rules for the JSON line:
- mrn: Extract the patient's Medical Record Number (MRN), Registration Number, ID, UID, or IP/OP number from the content. If not found, use "Not mentioned".
- age: Extract patient age from the content. If not found, use "unknown".
- sex: Extract patient sex/gender from the content. Use "M" for male, "F" for female. If not found, use "unknown".
- study: The imaging study that is currently indicated/requested (e.g., "CT Thorax", "MRI Brain", "USG Abdomen"). If not obvious from the content, use "Not mentioned".
- brief: A very short clinical summary using medical abbreviations. Example: "H/o fever and cough for 4 days, SOB for 2 days, K/c/o ILD, Now scan done to r/o infective exacerbation" or "C/o Giddiness for 15 days, slurred speech for 5 days, Right upper limb weakness for 2 days, K/c/o HTN/DM, Now scan done to r/o cerebellar infarct"`;

const SECONDARY_SYSTEM_INSTRUCTION = `You are an expert radiologist. When you receive a context, it is mostly about a patient and sometimes they might have been advised with any imaging modality. You analyse that info and then advise regarding that as an expert radiologist what to be seen in that specific imaging modality for that specific patient including various hypothetical imaging findings from common to less common for that patient condition in that specific imaging modality. suppose of you cant indentify thr specific imaging modality in thr given context, you yourself choose the appropriate imaging modality based on the specific conditions context`;

const SECONDARY_TRIGGER_PROMPT = `Here is the Clinical Profile generated from the patient's reports. Please analyze this profile according to your system instructions and provide the final output.`;

const CONFIG = {
  API_KEYS: getApiKeys(),
  GEMINI_MODEL: 'gemini-3.1-flash-lite', // Fast model locked
  TELEGRAM_TOKEN: process.env.TELEGRAM_BOT_TOKEN,
  ADMIN_ID: process.env.ADMIN_ID, // Restricts /users command & receives forwarded messages
  MEDIA_TIMEOUT_MS: 300000, // 5 minutes
  COMMANDS: ['.', '.1', '.2', '.3', '..', '..1', '..2', '..3', 'help', 'clear', 'status', 'users']
};

const GROUP_REPLY_FOOTER = `

━━━━━━━━━━━━━━━━━━━━━━
🖼️ *Go here to see the scan images:*
https://view.stradus.com/

🤖 *Copy-paste the clinical profile here to get suggestions regarding MRI protocols:*
https://ai.studio/apps/86a65a19-cf2f-46de-b4d0-9a941be83604

🔗 https://mri-protocols.vercel.app/

📚 *MRI protocol books*
https://notebooklm.google.com/notebook/467e8684-c512-488f-b1f7-3a450e344cd5`;

// ======================================================================
// 📊 DATA STORAGE, TIMEOUTS, USER TRACKING (In-Memory Only)
// ======================================================================
const chatMediaBuffers = new Map();
const chatTimeouts = new Map();
const registeredUsers = new Map(); // tracks users who messaged the bot since startup

// Automatically registers user data and forwards messages to the admin
async function trackAndForward(ctx) {
  const from = ctx.from;
  if (!from) return;
  const userId = String(from.id);
  const fullName = [from.first_name, from.last_name].filter(Boolean).join(' ');
  
  // Track user internally
  registeredUsers.set(userId, {
    username: from.username ? `@${from.username}` : 'No username',
    name: fullName || 'No name',
    lastSeen: new Date().toLocaleString()
  });

  // Forward message copy to the Administrator if the message is from another user
  const adminId = CONFIG.ADMIN_ID;
  if (adminId && userId !== String(adminId)) {
    try {
      await ctx.telegram.forwardMessage(adminId, ctx.chat.id, ctx.message.message_id);
    } catch (e) {
      console.error(`📡 Forwarding to Admin failed: ${e.message}`);
    }
  }
}

function getChatBuffer(chatId) {
  if (!chatMediaBuffers.has(chatId)) {
    chatMediaBuffers.set(chatId, []);
  }
  return chatMediaBuffers.get(chatId);
}

function clearChatBuffer(chatId) {
  if (chatTimeouts.has(chatId)) {
    clearTimeout(chatTimeouts.get(chatId));
    chatTimeouts.delete(chatId);
  }
  const items = chatMediaBuffers.get(chatId) || [];
  chatMediaBuffers.delete(chatId);
  return items;
}

function resetChatTimeout(chatId, ctx) {
  if (chatTimeouts.has(chatId)) {
    clearTimeout(chatTimeouts.get(chatId));
  }

  chatTimeouts.set(chatId, setTimeout(async () => {
    const cleared = clearChatBuffer(chatId);
    if (cleared.length > 0) {
      try {
        await ctx.reply(`⏰ *Buffer Timeout:* Your pending ${cleared.length} files were cleared due to inactivity. Please upload them again.`);
      } catch (e) {
        console.error('Timeout message error:', e.message);
      }
    }
  }, CONFIG.MEDIA_TIMEOUT_MS));
}

// ======================================================================
// 🛠️ HELPERS (Formatting, Downloader, Frame Extractor, Chunked Sender)
// ======================================================================

async function getTelegramFileAsBase64(ctx, fileId) {
  const fileLink = await ctx.telegram.getFileLink(fileId);
  const response = await axios.get(fileLink.href, { responseType: 'arraybuffer' });
  return Buffer.from(response.data).toString('base64');
}

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

    console.log(`Smart Frame Extraction: Target ${targetFps}fps`);

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
        try { if (fs.existsSync(inputPath)) fs.unlinkSync(inputPath); } catch (e) {}
        reject(err);
      })
      .run();
  });
}

function parseJsonFromResponse(responseText) {
  const jsonMatch = responseText.match(/<<JSON>>(.*?)<<JSON>>/s);
  if (jsonMatch && jsonMatch[1]) {
    try {
      return JSON.parse(jsonMatch[1].trim());
    } catch (e) {
      console.log(`⚠️ Failed to parse JSON block: ${e.message}`);
      return null;
    }
  }
  return null;
}

function stripJsonFromResponse(responseText) {
  return responseText.replace(/\n*<<JSON>>.*?<<JSON>>\n*/s, '').trim();
}

function formatJsonBlock(jsonData) {
  if (!jsonData) return '';
  const mrn = jsonData.mrn || 'Not mentioned';
  const age = jsonData.age || 'unknown';
  const sex = jsonData.sex || 'unknown';
  const study = jsonData.study || 'Not mentioned';
  const brief = jsonData.brief || '';
  return `\n\n📋 *Quick Reference:*\n• MRN/Reg No: ${mrn}\n• Age: ${age}\n• Sex: ${sex}\n• Study: ${study}\n• Brief: ${brief}`;
}

async function sendSafeMessage(ctx, text) {
  const MAX_LENGTH = 4000;
  const chunks = [];
  let remainingText = text;

  while (remainingText.length > 0) {
    if (remainingText.length <= MAX_LENGTH) {
      chunks.push(remainingText);
      break;
    }
    let splitIndex = remainingText.lastIndexOf('\n', MAX_LENGTH);
    if (splitIndex === -1) splitIndex = MAX_LENGTH;

    chunks.push(remainingText.substring(0, splitIndex));
    remainingText = remainingText.substring(splitIndex).trim();
  }

  let lastSentMsg;
  for (const chunk of chunks) {
    try {
      lastSentMsg = await ctx.reply(chunk, { parse_mode: 'Markdown' });
    } catch (e) {
      lastSentMsg = await ctx.reply(chunk);
    }
  }
  return lastSentMsg;
}

// ======================================================================
// 🧠 GEMINI API FAILOVER LOOP (Locked to fast model config)
// ======================================================================
async function generateGeminiContent(requestContent, systemInstruction) {
  const keys = CONFIG.API_KEYS;
  if (keys.length === 0) {
    throw new Error('No API keys configured! Check GEMINI_API_KEYS variable.');
  }

  let lastErrorMsg = '';
  const modelName = CONFIG.GEMINI_MODEL;

  for (let i = 0; i < keys.length; i++) {
    try {
      if (i > 0) {
        console.log(`⚠️ Failover: Trying Backup API Key #${i + 1}...`);
        await new Promise(r => setTimeout(r, 1500));
      }

      const genAI = new GoogleGenerativeAI(keys[i]);
      const modelConfig = { model: modelName };
      
      if (systemInstruction) {
        modelConfig.systemInstruction = systemInstruction;
      }

      const model = genAI.getGenerativeModel(modelConfig);
      const result = await model.generateContent(requestContent);
      const responseText = result.response.text();

      if (!responseText) {
        throw new Error("Received empty response from API");
      }

      return responseText;

    } catch (error) {
      lastErrorMsg = error.message;
      console.error(`❌ Key #${i + 1} failed:`, error.message);
    }
  }

  throw new Error(`All API keys failed. Last error: ${lastErrorMsg}`);
}

// ======================================================================
// 🚀 PIPELINE PROCESSOR
// ======================================================================
async function processMedia(ctx, chatId, mediaFiles, targetFps = 3, isSecondaryMode = false) {
  try {
    const counts = { images: 0, pdfs: 0, audio: 0, video: 0, texts: 0 };
    const captions = [];
    const textContents = [];
    const binaryMedia = [];

    // Smart frame extraction for videos
    const processedMedia = [];
    for (const m of mediaFiles) {
      if (m.type === 'video') {
        try {
          const videoBuffer = Buffer.from(m.data, 'base64');
          const frames = await extractFramesFromVideo(videoBuffer, targetFps);
          frames.forEach(frameData => {
            processedMedia.push({
              type: 'image',
              data: frameData,
              mimeType: 'image/jpeg',
              caption: m.caption ? `[Frame from video] ${m.caption}` : '[Frame from video]'
            });
          });
        } catch (err) {
          console.error(`Video extraction failed, treating as standard video: ${err.message}`);
          processedMedia.push(m);
        }
      } else {
        processedMedia.push(m);
      }
    }

    processedMedia.forEach(m => {
      if (m.type === 'image') {
        counts.images++;
        binaryMedia.push(m);
        if (m.caption) captions.push(`[Image caption]: ${m.caption}`);
      } else if (m.type === 'pdf') {
        counts.pdfs++;
        binaryMedia.push(m);
        if (m.caption) captions.push(`[PDF caption]: ${m.caption}`);
      } else if (m.type === 'audio' || m.type === 'voice') {
        counts.audio++;
        binaryMedia.push(m);
        if (m.caption) captions.push(`[Audio caption]: ${m.caption}`);
      } else if (m.type === 'text') {
        counts.texts++;
        textContents.push(`[Text note]: ${m.content}`);
      }
    });

    const contentParts = binaryMedia.map(m => ({
      inlineData: { data: m.data, mimeType: m.mimeType }
    }));

    const allOriginalText = [...captions, ...textContents];
    let promptParts = [];
    if (counts.images > 0) promptParts.push(`${counts.images} image(s)`);
    if (counts.pdfs > 0) promptParts.push(`${counts.pdfs} PDF document(s)`);
    if (counts.audio > 0) promptParts.push(`${counts.audio} audio/voice recording(s)`);

    let promptText = `Analyze these ${promptParts.join(', ')} along with the following additional text notes/context, and generate the Clinical Profile.

=== ADDITIONAL TEXT NOTES ===
${allOriginalText.join('\n\n')}
=== END OF TEXT NOTES ===

For audio files, transcribe the content first.`;

    const currentDate = new Date().toLocaleDateString('en-GB', {
      day: 'numeric', month: 'long', year: 'numeric'
    });
    promptText += `\n\n⚠️ CRITICAL INSTRUCTION REGARDING DATES: 
Today's current date is ${currentDate}. Please pay extremely close attention to the dates printed or handwritten on the medical reports. You MUST extract and transcribe the year EXACTLY as it appears in the images/documents. Do NOT let your training biases replace the current year with past years.`;

    const requestContent = contentParts.length > 0 ? [promptText, ...contentParts] : [promptText];

    // STEP 1: Clinical Profile Compile
    const rawPrimaryResponse = await generateGeminiContent(requestContent, PRIMARY_SYSTEM_INSTRUCTION);
    const jsonData = parseJsonFromResponse(rawPrimaryResponse);
    const primaryResponseText = stripJsonFromResponse(rawPrimaryResponse);

    if (isSecondaryMode) {
      let step1Text = `📝 *Clinical Profile (Step 1):*\n\n${primaryResponseText}`;
      if (jsonData) step1Text += formatJsonBlock(jsonData);
      step1Text += GROUP_REPLY_FOOTER;

      await sendSafeMessage(ctx, step1Text);

      // STEP 2: Secondary modality guidelines
      const secondaryPrompt = `${SECONDARY_TRIGGER_PROMPT}\n\n=== CLINICAL PROFILE ===\n${primaryResponseText}\n=== END PROFILE ===`;
      const secondaryResponseText = await generateGeminiContent([secondaryPrompt], SECONDARY_SYSTEM_INSTRUCTION);

      let step2Text = `🧠 *Secondary Analysis (Step 2):*\n\n${secondaryResponseText}`;
      step2Text += GROUP_REPLY_FOOTER;

      await sendSafeMessage(ctx, step2Text);
      return;
    }

    let finalResponseText = primaryResponseText;
    if (jsonData) {
      finalResponseText += formatJsonBlock(jsonData);
    }
    finalResponseText += GROUP_REPLY_FOOTER;

    await sendSafeMessage(ctx, finalResponseText);

  } catch (error) {
    console.error('Execution pipeline error:', error);
    await ctx.reply(`❌ Error processing request: ${error.message}`);
  }
}

// ======================================================================
// 📱 TELEGRAM BOT COMMANDS & HANDLERS
// ======================================================================
if (!CONFIG.TELEGRAM_TOKEN) {
  console.error("❌ No TELEGRAM_BOT_TOKEN defined in environment!");
  process.exit(1);
}

const bot = new Telegraf(CONFIG.TELEGRAM_TOKEN);

bot.command('start', async (ctx) => {
  await trackAndForward(ctx);
  await ctx.reply(`🏥 *Medical Clinical Profile Bot Ready*

*Supported Files:*
📷 Images, 📄 PDFs, 🎤 Voice, 🎵 Audio, 🎬 Videos

*Manual Processing Commands:*
• *.* - Standard Clinical Profile (Smart 3 FPS)
• *..* - Secondary Modality Chained Analysis (Profile + Advice)
• *.1 / ..1* - Smart 1 FPS Extraction
• *.2 / ..2* - Smart 2 FPS Extraction
• *clear* - Clear buffered documents
• *status* - View pending items in queue`, { parse_mode: 'Markdown' });
});

bot.command('clear', async (ctx) => {
  await trackAndForward(ctx);
  const chatId = ctx.chat.id;
  const cleared = clearChatBuffer(chatId);
  await ctx.reply(`🗑️ Cleared ${cleared.length} items from your buffer.`);
});

bot.command('status', async (ctx) => {
  await trackAndForward(ctx);
  const chatId = ctx.chat.id;
  const buffer = getChatBuffer(chatId);
  const counts = { images: 0, pdfs: 0, audio: 0, video: 0, texts: 0 };

  buffer.forEach(b => {
    if (b.type === 'image') counts.images++;
    else if (b.type === 'pdf') counts.pdfs++;
    else if (b.type === 'audio' || b.type === 'voice') counts.audio++;
    else if (b.type === 'video') counts.video++;
    else if (b.type === 'text') counts.texts++;
  });

  const text = `📊 *Buffer Status:*
📷 Images: ${counts.images}
📄 PDFs: ${counts.pdfs}
🎵 Audio/Voice: ${counts.audio}
🎬 Videos: ${counts.video}
📝 Text Notes: ${counts.texts}
━━━━━━━━━━
📦 Total buffered: ${buffer.length} / 20 items max`;
  await ctx.reply(text, { parse_mode: 'Markdown' });
});

bot.command('users', async (ctx) => {
  await trackAndForward(ctx);
  const userId = String(ctx.from.id);
  const adminId = CONFIG.ADMIN_ID;

  if (adminId && userId !== String(adminId)) {
    return ctx.reply("⚠️ Access denied. You are not authorized to view bot statistics.");
  }

  if (registeredUsers.size === 0) {
    return ctx.reply("👥 No active users recorded since the last boot.");
  }

  let userListText = `👥 *Unique Users since last startup (${registeredUsers.size}):*\n\n`;
  registeredUsers.forEach((data, id) => {
    userListText += `• *ID:* \`${id}\`\n  *Name:* ${data.name}\n  *Username:* ${data.username}\n  *Last Active:* ${data.lastSeen}\n\n`;
  });

  await sendSafeMessage(ctx, userListText);
});

// Media Queue Handlers
const registerMediaItem = async (ctx, type, fileId, mimeType, captionText) => {
  await trackAndForward(ctx);
  const chatId = ctx.chat.id;

  try {
    const loadingMsg = await ctx.reply(`📥 Downloading file to in-memory buffer...`);
    const base64Data = await getTelegramFileAsBase64(ctx, fileId);
    ctx.telegram.deleteMessage(chatId, loadingMsg.message_id).catch(() => {});

    const buffer = getChatBuffer(chatId);
    if (buffer.length >= 20) {
      await ctx.reply(`⚠️ Buffer is full. Clear using /clear or process using *.*`);
      return;
    }

    buffer.push({
      type: type,
      data: base64Data,
      mimeType: mimeType,
      caption: captionText || ''
    });

    resetChatTimeout(chatId, ctx);
    await ctx.reply(`📎 Added ${type.toUpperCase()} to queue. Queue count: *${buffer.length}*`, { parse_mode: 'Markdown' });

  } catch (error) {
    console.error('Buffer queue error:', error);
    await ctx.reply(`❌ Failed to buffer file.`);
  }
};

bot.on(message('photo'), ctx => {
  const photo = ctx.message.photo;
  const fileId = photo[photo.length - 1].file_id;
  registerMediaItem(ctx, 'image', fileId, 'image/jpeg', ctx.message.caption);
});

bot.on(message('document'), ctx => {
  const doc = ctx.message.document;
  if (!doc.mime_type || !doc.mime_type.includes('pdf')) {
    return ctx.reply("⚠️ Only PDF documents are supported!");
  }
  registerMediaItem(ctx, 'pdf', doc.file_id, 'application/pdf', ctx.message.caption);
});

bot.on(message('video'), ctx => {
  const vid = ctx.message.video;
  registerMediaItem(ctx, 'video', vid.file_id, vid.mime_type || 'video/mp4', ctx.message.caption);
});

bot.on(message('voice'), ctx => {
  const voice = ctx.message.voice;
  registerMediaItem(ctx, 'voice', voice.file_id, voice.mime_type || 'audio/ogg', ctx.message.caption);
});

bot.on(message('audio'), ctx => {
  const audio = ctx.message.audio;
  registerMediaItem(ctx, 'audio', audio.file_id, audio.mime_type || 'audio/mpeg', ctx.message.caption);
});

bot.on(message('text'), async (ctx) => {
  await trackAndForward(ctx);
  const text = ctx.message.text.trim();
  const chatId = ctx.chat.id;

  const isPrimaryTrigger = /^(\.|(\.[1-3]))$/.test(text);
  const isSecondaryTrigger = /^(\.\.|(\.\.[1-3]))$/.test(text);

  if (isPrimaryTrigger || isSecondaryTrigger) {
    const mediaFiles = clearChatBuffer(chatId);
    if (mediaFiles.length === 0) {
      await ctx.reply("ℹ️ Buffer empty. Please upload some files or type some context first!");
      return;
    }

    const lastChar = text.slice(-1);
    let targetFps = 3;
    if (!isNaN(parseInt(lastChar))) {
      targetFps = parseInt(lastChar);
    }

    const mode = isSecondaryTrigger ? 'secondary' : 'primary';
    const label = isSecondaryTrigger ? 'Chained Secondary Analysis' : 'Clinical Profile';

    const statusMsg = await ctx.reply(`⏳ Running ${label} on ${mediaFiles.length} files (Smart ${targetFps} FPS)...`);

    try {
      await processMedia(ctx, chatId, mediaFiles, targetFps, isSecondaryTrigger);
      ctx.telegram.deleteMessage(chatId, statusMsg.message_id).catch(() => {});
    } catch (err) {
      await ctx.reply(`❌ Processing Failed: ${err.message}`);
    }
    return;
  }

  // Handle clinical text input added to buffer
  const buffer = getChatBuffer(chatId);
  buffer.push({
    type: 'text',
    content: text
  });
  resetChatTimeout(chatId, ctx);
  await ctx.reply(`📝 Text note added to buffer. Queue count: *${buffer.length}*`, { parse_mode: 'Markdown' });
});

// ======================================================================
// 🌐 WEB SERVER & BOT INITIALIZATION
// ======================================================================
const app = express();
const PORT = process.env.PORT || 3000;
app.get('/', (req, res) => res.send('Telegram Medical Profile Bot Server Running Active'));
app.get('/health', (req, res) => res.json({ status: 'healthy', database: 'none' }));
app.listen(PORT, () => console.log(`🌐 Web server active on port ${PORT}`));

// ======================================================================
// 🔄 SELF-PINGING KEEP-ALIVE SYSTEM (Keeps Render Free Tier Awake)
// ======================================================================
const RENDER_URL = process.env.RENDER_EXTERNAL_URL;
if (RENDER_URL) {
  console.log(`📡 Self-ping Keep-Alive registered for: ${RENDER_URL}`);
  
  // Ping the server's own public URL every 10 minutes to reset Render's 15-minute sleep timer
  setInterval(async () => {
    try {
      const pingUrl = `${RENDER_URL}/health`;
      const response = await axios.get(pingUrl);
      console.log(`📡 Self-ping complete: Status ${response.status} (Keep-Alive Reset Successful)`);
    } catch (error) {
      console.error(`📡 Self-ping error: ${error.message}`);
    }
  }, 10 * 60 * 1000); // 10 minutes
} else {
  console.log('⚠️ RENDER_EXTERNAL_URL is undefined. Internal self-ping is offline (Local environment).');
}

bot.launch(() => console.log('🚀 Telegram Bot Engine Active (MongoDB and replies disabled)'));

process.once('SIGINT', () => bot.stop('SIGINT'));
process.once('SIGTERM', () => bot.stop('SIGTERM'));
