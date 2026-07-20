// PenuX Follow-Up Automation — Google Apps Script
// Run order: autoObtainAllKeys() → runPenuXComplete() → then daily trigger auto-runs

// ─── CONFIGURATION ───────────────────────────────────────────────────────────
var PENUX_LABEL      = 'PenuX';
var SEARCH_DAYS      = 30;
var MY_NAME          = 'Netanel Stern';
var MY_PHONE         = '+972559708708';
var GEMINI_MODEL     = 'gemini-1.5-flash';
var GEMINI_PRO_MODEL = 'gemini-1.5-pro';

// ─── ENTRY POINT ─────────────────────────────────────────────────────────────
function runPenuXComplete() {
  autoObtainAllKeys();
  var geminiKey = getKey('GEMINI_API_KEY');
  if (!geminiKey) {
    setKey('GEMINI_API_KEY', 'YOUR_GEMINI_KEY_HERE');
    Logger.log('⚠️  Set your Gemini key via: setKey("GEMINI_API_KEY","sk-...")');
  }
  exportPenuXPDF();
  schedulePenuXCalendarEvents();
  schedulePenuXFollowUps();
  deleteAllTriggers();
  ScriptApp.newTrigger('schedulePenuXFollowUps')
    .timeBased().atHour(9).everyDays(1).create();
  Logger.log('✅ PenuX setup complete. Daily 9am trigger active.');
}

// ─── PDF EXPORT ──────────────────────────────────────────────────────────────
function exportPenuXPDF() {
  var threads = searchPenuXThreads();
  var doc = DocumentApp.create('PenuX Summary ' + new Date().toLocaleDateString('he-IL'));
  var body = doc.getBody();
  body.appendParagraph('PenuX Email Summary').setHeading(DocumentApp.ParagraphHeading.TITLE);

  threads.forEach(function(thread) {
    var msg    = thread.getMessages()[0];
    var subj   = msg.getSubject();
    var sender = msg.getFrom();
    var date   = msg.getDate();
    var plain  = msg.getPlainBody().substring(0, 300);
    var subjEn = translateSubject(subj);
    var summary = callGemini(
      'Summarize this email in 2 sentences (English):\nSubject: ' + subj + '\n' + plain
    );
    body.appendParagraph(subjEn || subj).setHeading(DocumentApp.ParagraphHeading.HEADING1);
    body.appendParagraph('From: ' + sender + '  |  ' + date.toLocaleDateString('he-IL'));
    body.appendParagraph(summary || plain);
    body.appendHorizontalRule();
  });

  doc.saveAndClose();
  var pdf = DriveApp.getFileById(doc.getId()).getAs('application/pdf');
  var folder = DriveApp.getRootFolder();
  folder.createFile(pdf);
  Logger.log('📄 PDF saved: ' + doc.getName());
}

// ─── CALENDAR SCHEDULING ─────────────────────────────────────────────────────
function schedulePenuXCalendarEvents() {
  var threads = searchPenuXThreads();
  var cal = CalendarApp.getDefaultCalendar();
  threads.forEach(function(thread) {
    var msg  = thread.getMessages()[0];
    var subj = msg.getSubject();
    var body = msg.getPlainBody().substring(0, 500);
    var resp = callGemini(
      'Extract meeting date/time from this email. Return JSON: {"date":"YYYY-MM-DD","time":"HH:MM","duration_min":60}. If none, return {}.\n' + body
    );
    var data = parseJson(resp);
    if (data && data.date && data.time) {
      var start = new Date(data.date + 'T' + data.time + ':00');
      var end   = new Date(start.getTime() + (data.duration_min || 60) * 60000);
      cal.createEvent('PenuX: ' + subj, start, end);
      Logger.log('📅 Event created: ' + subj + ' at ' + start);
    }
  });
}

// ─── FOLLOW-UP SCHEDULING ────────────────────────────────────────────────────
function schedulePenuXFollowUps() {
  var props   = PropertiesService.getScriptProperties();
  var threads = searchPenuXThreads();
  var today   = new Date();

  threads.forEach(function(thread) {
    var msg      = thread.getMessages()[0];
    var threadId = thread.getId();
    var sender   = msg.getFrom();
    var subj     = msg.getSubject();
    var body     = msg.getPlainBody().substring(0, 600);

    // Check OOO
    var oooResult = detectOOO(body);
    if (oooResult.isOOO) {
      var returnDate = gatherReturnDate(body, sender);
      props.setProperty('FU_' + threadId, JSON.stringify({
        email: sender, subject: subj,
        sendAfter: returnDate.toISOString(),
        returnDate: returnDate.toISOString(),
        draft: buildFollowUpDraft(subj, sender)
      }));
      Logger.log('🏖️  OOO detected for ' + sender + ', follow-up after ' + returnDate);
      return;
    }

    // LLM consensus
    var prompt = buildFollowUpPrompt(subj, body, sender);
    var results = callAllLLMs(prompt);
    var merged = mergeRecommendations(results);
    if (!merged) return;

    if (merged.should_followup) {
      var sendDate = new Date(today.getTime() + merged.days_to_wait * 86400000);
      props.setProperty('FU_' + threadId, JSON.stringify({
        email: sender, subject: subj,
        sendAfter: sendDate.toISOString(),
        draft: merged.suggested_message
      }));
      Logger.log('📬 Follow-up scheduled for ' + sender + ' in ' + merged.days_to_wait + ' days');
    }
  });

  sendDueFollowUps();
}

function sendDueFollowUps() {
  var props = PropertiesService.getScriptProperties();
  var all   = props.getProperties();
  var now   = new Date();
  Object.keys(all).forEach(function(key) {
    if (!key.startsWith('FU_')) return;
    var rec = parseJson(all[key]);
    if (!rec || !rec.sendAfter) return;
    if (new Date(rec.sendAfter) > now) return;
    var draft = rec.draft || '';
    if (draft.toLowerCase().indexOf('penux') === -1) {
      Logger.log('⏭️  Skipping (no "PenuX" in draft): ' + rec.email);
      props.deleteProperty(key);
      return;
    }
    GmailApp.sendEmail(rec.email, 'Re: ' + rec.subject, draft, {
      name: MY_NAME + ' | PenuX',
      replyTo: Session.getActiveUser().getEmail()
    });
    Logger.log('📨 Follow-up sent to ' + rec.email);
    props.deleteProperty(key);
  });
}

function buildFollowUpPrompt(subject, body, sender) {
  return 'You are an email assistant for ' + MY_NAME + ' (PenuX medical AI project).\n' +
    'Analyze this email thread and decide if a follow-up is needed.\n' +
    'Return JSON: {"should_followup":true,"days_to_wait":7,"suggested_message":"..."}\n' +
    'The suggested_message MUST mention the word PenuX.\n' +
    'Sender: ' + sender + '\nSubject: ' + subject + '\nBody: ' + body;
}

function buildFollowUpDraft(subject, sender) {
  return 'Hi,\n\nI wanted to follow up regarding "' + subject + '".\n' +
    'I am reaching out in the context of PenuX — our AI-driven medical platform.\n\n' +
    'Best regards,\n' + MY_NAME + '\n' + MY_PHONE;
}

// ─── MULTI-LLM CALLS ─────────────────────────────────────────────────────────
function callAllLLMs(prompt) {
  var results = [];
  results.push(callGemini(prompt));      // Gemini Flash
  results.push(callGeminiPro(prompt));   // Gemini Pro
  results.push(callOpenAI(prompt));
  results.push(callClaude(prompt));
  return results.map(function(r) { return parseJson(r); }).filter(Boolean);
}

function mergeRecommendations(results) {
  if (!results || results.length === 0) return null;
  var votes = results.filter(function(r) { return r.should_followup; }).length;
  var majority = votes >= Math.ceil(results.length / 2);
  var days = results.reduce(function(sum, r) {
    return sum + (parseInt(r.days_to_wait) || 7);
  }, 0) / results.length;
  var best = results.find(function(r) { return r.suggested_message; });
  return {
    should_followup: majority,
    days_to_wait: Math.round(days),
    suggested_message: best ? best.suggested_message : ''
  };
}

function callGemini(prompt) {
  return callGeminiModel(prompt, GEMINI_MODEL);
}

function callGeminiPro(prompt) {
  return callGeminiModel(prompt, GEMINI_PRO_MODEL);
}

function callGeminiModel(prompt, model) {
  var key = getKey('GEMINI_API_KEY');
  var token = ScriptApp.getOAuthToken();
  var url, headers, payload;
  if (key) {
    url = 'https://generativelanguage.googleapis.com/v1beta/models/' + model + ':generateContent?key=' + key;
    headers = {'Content-Type': 'application/json'};
  } else {
    url = 'https://generativelanguage.googleapis.com/v1beta/models/' + model + ':generateContent';
    headers = {'Content-Type': 'application/json', 'Authorization': 'Bearer ' + token};
  }
  payload = JSON.stringify({contents:[{parts:[{text: prompt}]}]});
  try {
    var resp = UrlFetchApp.fetch(url, {method:'post', headers:headers, payload:payload, muteHttpExceptions:true});
    var data = JSON.parse(resp.getContentText());
    return data.candidates && data.candidates[0] &&
      data.candidates[0].content.parts[0].text || '';
  } catch(e) {
    Logger.log('Gemini error: ' + e);
    return '';
  }
}

function callOpenAI(prompt) {
  var key = getKey('OPENAI_API_KEY');
  if (!key) return '';
  try {
    var resp = UrlFetchApp.fetch('https://api.openai.com/v1/chat/completions', {
      method: 'post',
      headers: {'Content-Type':'application/json','Authorization':'Bearer '+key},
      payload: JSON.stringify({model:'gpt-4o-mini',messages:[{role:'user',content:prompt}]}),
      muteHttpExceptions: true
    });
    var data = JSON.parse(resp.getContentText());
    return data.choices && data.choices[0] && data.choices[0].message.content || '';
  } catch(e) { Logger.log('OpenAI error: '+e); return ''; }
}

function callClaude(prompt) {
  var key = getKey('ANTHROPIC_API_KEY');
  if (!key) return '';
  try {
    var resp = UrlFetchApp.fetch('https://api.anthropic.com/v1/messages', {
      method: 'post',
      headers: {
        'Content-Type':'application/json',
        'x-api-key': key,
        'anthropic-version': '2023-06-01'
      },
      payload: JSON.stringify({
        model: 'claude-haiku-4-5-20251001',
        max_tokens: 512,
        messages: [{role:'user',content:prompt}]
      }),
      muteHttpExceptions: true
    });
    var data = JSON.parse(resp.getContentText());
    return data.content && data.content[0] && data.content[0].text || '';
  } catch(e) { Logger.log('Claude error: '+e); return ''; }
}

// ─── OOO DETECTION ───────────────────────────────────────────────────────────
function detectOOO(body) {
  var oooKeywords = [
    'out of office','on vacation','on leave','away from office',
    'will be back','returning on','auto-reply','automatic reply',
    'חופשה','לא במשרד','אחזור','חזרה','הודעה אוטומטית'
  ];
  var lower = body.toLowerCase();
  var found = oooKeywords.some(function(kw) { return lower.indexOf(kw) !== -1; });
  return { isOOO: found };
}

// ─── RETURN DATE GATHERING ───────────────────────────────────────────────────
function gatherReturnDate(oooText, senderEmail) {
  // Layer 1: regex
  var d = extractDateByRegex(oooText);
  if (d) {
    logReturnDate(senderEmail, d, 'regex');
    return d;
  }
  // Layer 2: Gemini
  var resp = callGemini(
    'Extract the return-to-office date from this OOO message. Return only YYYY-MM-DD or empty string.\n' + oooText
  );
  if (resp && /^\d{4}-\d{2}-\d{2}$/.test(resp.trim())) {
    var gd = new Date(resp.trim());
    logReturnDate(senderEmail, gd, 'gemini');
    return gd;
  }
  // Layer 3: fallback
  var fallback = new Date(Date.now() + 7 * 86400000);
  logReturnDate(senderEmail, fallback, 'fallback');
  return fallback;
}

function extractDateByRegex(text) {
  var months = {
    jan:1,feb:2,mar:3,apr:4,may:5,jun:6,
    jul:7,aug:8,sep:9,oct:10,nov:11,dec:12,
    january:1,february:2,march:3,april:4,june:6,
    july:7,august:8,september:9,october:10,november:11,december:12
  };
  var patterns = [
    /(\d{4})-(\d{2})-(\d{2})/,
    /(\d{1,2})\/(\d{1,2})\/(\d{4})/,
    /(\d{1,2})\.(\d{1,2})\.(\d{4})/,
    /(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\s+(\d{1,2})/i,
    /back\s+on\s+(\w+)\s+(\d{1,2})/i
  ];
  for (var i = 0; i < patterns.length; i++) {
    var m = text.match(patterns[i]);
    if (m) {
      if (i === 0) return new Date(m[1]+'-'+m[2]+'-'+m[3]);
      if (i === 1) return new Date(m[3]+'-'+m[2]+'-'+m[1]);
      if (i === 2) return new Date(m[3]+'-'+m[2]+'-'+m[1]);
      if (i === 3 && months[m[1].toLowerCase()]) {
        var mn = months[m[1].toLowerCase()];
        var dateResult = new Date(new Date().getFullYear(), mn-1, parseInt(m[2]));
        return dateResult;
      }
    }
  }
  return null;
}

function logReturnDate(email, date, source) {
  Logger.log('Return date for ' + email + ': ' + date.toISOString().split('T')[0] + ' (source: ' + source + ')');
}

// ─── TRANSLATION ─────────────────────────────────────────────────────────────
function translateSubject(subject) {
  var translateKey = getKey('GOOGLE_TRANSLATE_API_KEY');
  if (translateKey) {
    try {
      var url = 'https://translation.googleapis.com/language/translate/v2?key=' + translateKey;
      var resp = UrlFetchApp.fetch(url, {
        method: 'post',
        contentType: 'application/json',
        payload: JSON.stringify({q: subject, target: 'en'})
      });
      var data = JSON.parse(resp.getContentText());
      return data.data.translations[0].translatedText;
    } catch(e) { Logger.log('Translate API error: '+e); }
  }
  // Fallback: Gemini
  var resp2 = callGemini('Translate to English (return only translation): ' + subject);
  return resp2 || subject;
}

// ─── SEARCH ──────────────────────────────────────────────────────────────────
function searchPenuXThreads() {
  var query = 'label:' + PENUX_LABEL + ' newer_than:' + SEARCH_DAYS + 'd';
  return GmailApp.search(query, 0, 50);
}

// ─── AUTO-OBTAIN ALL KEYS ────────────────────────────────────────────────────
function autoObtainAllKeys() {
  Logger.log('🔑 Auto-obtaining API keys...');

  // 1. Google Translate key (fully automatic)
  autoObtainGoogleTranslateKey();

  // 2. Cohere trial key
  autoObtainCohereKey();

  // 3. Manual instructions for OpenAI + Anthropic
  Logger.log('─────────────────────────────────────────────');
  Logger.log('👉 OpenAI key: https://platform.openai.com/api-keys');
  Logger.log('   After obtaining: setKey("OPENAI_API_KEY","sk-...")');
  Logger.log('👉 Anthropic key: https://console.anthropic.com/settings/keys');
  Logger.log('   After obtaining: setKey("ANTHROPIC_API_KEY","sk-ant-...")');
  Logger.log('─────────────────────────────────────────────');
}

function autoObtainGoogleTranslateKey() {
  if (getKey('GOOGLE_TRANSLATE_API_KEY')) {
    Logger.log('✅ Google Translate key already set.');
    return;
  }
  try {
    var token = ScriptApp.getOAuthToken();
    var projectId = getGcpProjectId();
    if (!projectId) { Logger.log('⚠️  Could not determine GCP project ID.'); return; }

    // Enable APIs
    ['translate.googleapis.com','apikeys.googleapis.com'].forEach(function(api) {
      UrlFetchApp.fetch(
        'https://serviceusage.googleapis.com/v1/projects/' + projectId + '/services/' + api + ':enable',
        {method:'post', headers:{'Authorization':'Bearer '+token}, muteHttpExceptions:true}
      );
    });
    Utilities.sleep(3000);

    // Create API key
    var createResp = UrlFetchApp.fetch(
      'https://apikeys.googleapis.com/v2/projects/' + projectId + '/locations/global/keys',
      {
        method: 'post',
        headers: {'Authorization':'Bearer '+token,'Content-Type':'application/json'},
        payload: JSON.stringify({
          displayName: 'PenuX Translate',
          restrictions: {apiTargets:[{service:'translate.googleapis.com'}]}
        }),
        muteHttpExceptions: true
      }
    );
    var opName = JSON.parse(createResp.getContentText()).name;
    if (!opName) { Logger.log('⚠️  Could not create key, op name missing.'); return; }

    // Poll operation
    var apiKey = pollGoogleOperation(opName, token);
    if (apiKey) {
      setKey('GOOGLE_TRANSLATE_API_KEY', apiKey);
      Logger.log('✅ Google Translate API key created and saved.');
    }
  } catch(e) {
    Logger.log('⚠️  Google Translate auto-key failed: ' + e);
  }
}

function pollGoogleOperation(opName, token) {
  var base = 'https://apikeys.googleapis.com/v2/';
  var name = opName.replace(/^\//, '');
  for (var i = 0; i < 10; i++) {
    Utilities.sleep(2000);
    var r = UrlFetchApp.fetch(base + name, {
      headers: {'Authorization':'Bearer '+token},
      muteHttpExceptions: true
    });
    var data = JSON.parse(r.getContentText());
    if (data.done && data.response && data.response.keyString) {
      return data.response.keyString;
    }
  }
  return null;
}

function getGcpProjectId() {
  try {
    var r = UrlFetchApp.fetch(
      'https://www.googleapis.com/oauth2/v1/userinfo?alt=json',
      {headers:{'Authorization':'Bearer '+ScriptApp.getOAuthToken()}, muteHttpExceptions:true}
    );
    // Use Apps Script project number as proxy
    var id = ScriptApp.getScriptId();
    return id;
  } catch(e) { return null; }
}

function autoObtainCohereKey() {
  if (getKey('COHERE_API_KEY')) {
    Logger.log('✅ Cohere key already set.');
    return;
  }
  try {
    var resp = UrlFetchApp.fetch('https://api.cohere.ai/v1/users', {
      method: 'post',
      headers: {'Content-Type':'application/json'},
      payload: JSON.stringify({email: Session.getActiveUser().getEmail()}),
      muteHttpExceptions: true
    });
    Logger.log('Cohere signup response ' + resp.getResponseCode() + ': ' + resp.getContentText().substring(0,200));
    var data = JSON.parse(resp.getContentText());
    if (data && data.api_key) {
      setKey('COHERE_API_KEY', data.api_key);
      Logger.log('✅ Cohere trial key saved.');
    } else {
      Logger.log('👉 Cohere manual key: https://dashboard.cohere.com/api-keys');
      Logger.log('   After obtaining: setKey("COHERE_API_KEY","...")');
    }
  } catch(e) {
    Logger.log('⚠️  Cohere auto-key failed: ' + e);
    Logger.log('👉 Cohere manual key: https://dashboard.cohere.com/api-keys');
  }
}

// ─── UTILITIES ───────────────────────────────────────────────────────────────
function parseJson(text) {
  if (!text) return null;
  var s = text.replace(/```json\s*/gi,'').replace(/```\s*/g,'').trim();
  try { return JSON.parse(s); } catch(e) {
    var m = s.match(/\{[\s\S]*\}/);
    if (m) try { return JSON.parse(m[0]); } catch(e2) {}
    return null;
  }
}

function deleteAllTriggers() {
  ScriptApp.getProjectTriggers().forEach(function(t) { ScriptApp.deleteTrigger(t); });
  Logger.log('🗑️  All triggers deleted.');
}

function listScheduledFollowUps() {
  var props = PropertiesService.getScriptProperties().getProperties();
  Logger.log('=== Scheduled Follow-Ups ===');
  Object.keys(props).filter(function(k){ return k.startsWith('FU_'); }).forEach(function(k) {
    var rec = parseJson(props[k]);
    if (rec) Logger.log(rec.email + ' → ' + rec.sendAfter + (rec.returnDate ? ' (return: '+rec.returnDate+')' : ''));
  });
}

function debugAllLLMs() {
  var threads = GmailApp.search('label:' + PENUX_LABEL, 0, 1);
  if (!threads.length) { Logger.log('No PenuX threads found.'); return; }
  var msg    = threads[0].getMessages()[0];
  var prompt = buildFollowUpPrompt(msg.getSubject(), msg.getPlainBody().substring(0,300), msg.getFrom());
  Logger.log('=== Gemini Flash ===\n' + callGemini(prompt));
  Logger.log('=== Gemini Pro ===\n'   + callGeminiPro(prompt));
  Logger.log('=== OpenAI ===\n'       + callOpenAI(prompt));
  Logger.log('=== Claude ===\n'       + callClaude(prompt));
}

// ─── KEY HELPERS ─────────────────────────────────────────────────────────────
function setKey(name, value) {
  PropertiesService.getScriptProperties().setProperty(name, value);
  Logger.log('Saved: ' + name);
}

function getKey(name) {
  return PropertiesService.getScriptProperties().getProperty(name);
}

function deleteKey(name) {
  PropertiesService.getScriptProperties().deleteProperty(name);
  Logger.log('Deleted: ' + name);
}
