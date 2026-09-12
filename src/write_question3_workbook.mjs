/** Fill an official result3 template copy. Requires a validated full-year payload. */
import fs from 'node:fs/promises';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { FileBlob, SpreadsheetFile } from '@oai/artifact-tool';

const root = path.dirname(path.dirname(fileURLToPath(import.meta.url)));
const folder = path.resolve(process.argv[2] ?? path.join(root, 'results/q3_three_day'));
const payload = JSON.parse(await fs.readFile(path.join(folder, 'workbook_payload.json'), 'utf8'));
if (!payload.annual_complete || payload.plans.length !== 334 || payload.batteries.length !== 2004 || payload.labels.length !== 144 || payload.adjustments.length !== 334) {
  throw new Error('Incomplete annual payload; refusing to write result3.xlsx');
}
const outputPath = path.join(folder, 'result3.xlsx');
try { await fs.access(outputPath); throw new Error('Output already exists'); }
catch (error) { if (error.code !== 'ENOENT') throw error; }
const workbook = await SpreadsheetFile.importXlsx(await FileBlob.load(path.join(root, '附件/附件5/result3.xlsx')));
const plan = workbook.worksheets.getItem('计划购电量');
const adjusted = workbook.worksheets.getItem('调整购电量');
const battery = workbook.worksheets.getItem('充放电量');
const emergency = workbook.worksheets.getItem('紧急购电量');
const serial = value => (Date.parse(`${value}T00:00:00Z`) - Date.UTC(1899, 11, 30))/86400000;
const dates = rows => rows.map(row => [serial(row[0]), ...row.slice(1)]);

// Clear only the template's example body in the OUTPUT COPY, preserving headers.
battery.getRange('A2:F20').clear({applyTo:'contents'});
emergency.getRange('A2:C11').clear({applyTo:'contents'});
plan.getRange('B1:EO1').values = [payload.labels];
plan.getRange('A2:EQ335').values = dates(payload.plans);
adjusted.getRange('B1:EO1').values = [payload.labels];
adjusted.getRange('A2:EQ335').values = dates(payload.adjustments);
adjusted.getRange('B2:EQ335').format.numberFormat = '0.0000';
battery.getRange('A2:F2005').values = dates(payload.batteries);
emergency.getRange(`A2:C${payload.emergencies.length+1}`).values = dates(payload.emergencies);
for (const [sheet, lastRow] of [[plan, 335], [adjusted, 335], [battery, 2005], [emergency, payload.emergencies.length+1]]) {
  sheet.getRange(`A2:A${lastRow}`).format.numberFormat = 'yyyy-mm-dd';
}
// Preserve template columns and headers; use display precision only.
plan.getRange('B2:EQ335').format.numberFormat = '0.0000';
battery.getRange('C2:D2005').format.numberFormat = '0.0000';
battery.getRange('F2:F2005').format.numberFormat = '0.0000';
emergency.getRange(`C2:C${payload.emergencies.length+1}`).format.numberFormat = '0.0000';
workbook.recalculate();
for (const [sheet, range] of [['计划购电量','A1:D4'],['充放电量','A1:F8'],['紧急购电量','A1:C5']]) {
  console.log((await workbook.inspect({kind:'table', range:`${sheet}!${range}`, tableMaxRows:8, tableMaxCols:6, maxChars:1800})).ndjson);
}
console.log((await workbook.inspect({kind:'match', searchTerm:'#REF!|#DIV/0!|#VALUE!|#NAME\\?|#NUM!',
  options:{useRegex:true,maxResults:10}})).ndjson);
// Preview is opt-in and for layout QA only; normal competition runs save no images.
if (process.env.Q3_PREVIEW_DIR) {
  for (const [sheet, range] of [['计划购电量','A1:D5'],['充放电量','A1:F8'],['紧急购电量','A1:C6']]) {
    const preview = await workbook.render({sheetName:sheet, range, scale:1.5});
    await fs.writeFile(path.join(process.env.Q3_PREVIEW_DIR, `${sheet}.png`), new Uint8Array(await preview.arrayBuffer()));
  }
}
await (await SpreadsheetFile.exportXlsx(workbook)).save(outputPath);
console.log(outputPath);
