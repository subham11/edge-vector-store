// Prompt templates for Gemma 2 2B RAG conversation

export const SYSTEM_PROMPT = `You are a helpful agricultural assistant for Indian farmers. You have access to a database of farmer records from across India. When answering questions, use the provided context from the database to give accurate, specific answers. Always be helpful, clear, and concise.`;

export function buildRAGPrompt(
  userQuery: string,
  context: Array<{ id: string; distance: number; payload: Record<string, any> }>,
): string {
  const contextStr = context
    .map((doc, i) => {
      const p = doc.payload;
      return (
        `[${i + 1}] Farmer: ${p.farmerName}, State: ${p.state}, ` +
        `Crop: ${p.crop}, Soil: ${p.soilType}, Season: ${p.season}, ` +
        `Area: ${p.areaAcres} acres, Irrigation: ${p.irrigationType}, ` +
        `Weather: ${p.weather}, Daily Water: ${p.dailyWaterLiters}L, ` +
        `Fertilizers: ${(p.fertilizers || []).join(', ')}, ` +
        `Warnings: ${(p.warnings || []).join(', ')}`
      );
    })
    .join('\n');

  return (
    `<start_of_turn>user\n` +
    `Context from farmer database:\n${contextStr}\n\n` +
    `Question: ${userQuery}\n` +
    `<end_of_turn>\n` +
    `<start_of_turn>model\n`
  );
}

export function buildConversationPrompt(
  history: Array<{ role: 'user' | 'assistant'; content: string }>,
  currentQuery: string,
  context: Array<{ id: string; distance: number; payload: Record<string, any> }>,
): string {
  let prompt = `<start_of_turn>user\n${SYSTEM_PROMPT}<end_of_turn>\n`;

  // Add conversation history (keep last 4 turns to fit context window)
  const recent = history.slice(-4);
  for (const msg of recent) {
    if (msg.role === 'user') {
      prompt += `<start_of_turn>user\n${msg.content}<end_of_turn>\n`;
    } else {
      prompt += `<start_of_turn>model\n${msg.content}<end_of_turn>\n`;
    }
  }

  // Add current query with RAG context
  prompt += buildRAGPrompt(currentQuery, context);

  return prompt;
}
