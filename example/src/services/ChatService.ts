// ChatService — RAG pipeline connecting QueryEmbedder + VectorSearch + Gemma 2B

import RNFS from 'react-native-fs';
import { initLlama, type LlamaContext } from 'llama.rn';
import { VectorSearchService } from './VectorSearchService';
import { QueryEmbedder } from './QueryEmbedder';
import { buildConversationPrompt } from './PromptTemplates';

export interface ChatMessage {
  role: 'user' | 'assistant';
  content: string;
}

export class ChatService {
  private vectorSearch: VectorSearchService;
  private embedder: QueryEmbedder;
  private llama: LlamaContext | null = null;
  private history: ChatMessage[] = [];
  private useFallback = false;

  constructor() {
    this.vectorSearch = new VectorSearchService();
    this.embedder = new QueryEmbedder();
  }

  async initialize(onStatus?: (msg: string) => void): Promise<void> {
    // Initialize vector store
    await this.vectorSearch.initialize(onStatus);

    // Initialize embedding model
    await this.embedder.initialize(onStatus);

    // Initialize Gemma 2B via llama.cpp
    try {
      onStatus?.('Loading Gemma 2B model...');

      const modelCandidates = [
        `${RNFS.DocumentDirectoryPath}/gemma-2-2b-it-Q4_K_M.gguf`,
        `${RNFS.DocumentDirectoryPath}/gemma-2-2b-it.gguf`,
        `${RNFS.MainBundlePath}/gemma-2-2b-it-Q4_K_M.gguf`,
      ];

      let modelPath: string | null = null;
      for (const path of modelCandidates) {
        if (await RNFS.exists(path)) {
          modelPath = path;
          break;
        }
      }

      if (modelPath) {
        this.llama = await initLlama({
          model: modelPath,
          n_ctx: 2048,
          n_threads: 4,
        });
        onStatus?.('Gemma 2B ready');
      } else {
        onStatus?.(
          'Gemma model not found — using template-based responses',
        );
        this.useFallback = true;
      }
    } catch (e) {
      onStatus?.(`LLM unavailable: ${e} — using template-based responses`);
      this.useFallback = true;
    }
  }

  async chat(userMessage: string): Promise<string> {
    // 1. Embed the query
    const queryVector = await this.embedder.embed(userMessage);

    // 2. Search for relevant farmer records
    const results = await this.vectorSearch.search(
      Array.from(queryVector),
      5,
    );

    // 3. Build prompt with context
    const prompt = buildConversationPrompt(
      this.history,
      userMessage,
      results,
    );

    // 4. Generate response
    let response: string;
    if (this.llama && !this.useFallback) {
      const completion = await this.llama.completion({
        prompt,
        n_predict: 512,
        temperature: 0.7,
        top_p: 0.9,
        stop: ['<end_of_turn>'],
      });
      response = (completion as any).text?.trim() || this.buildFallbackResponse(results);
    } else {
      response = this.buildFallbackResponse(results);
    }

    // 5. Update history
    this.history.push({ role: 'user', content: userMessage });
    this.history.push({ role: 'assistant', content: response });

    // Keep history manageable
    if (this.history.length > 20) {
      this.history = this.history.slice(-10);
    }

    return response;
  }

  private buildFallbackResponse(
    results: Array<{ id: string; distance: number; payload?: Record<string, any> }>,
  ): string {
    if (results.length === 0) {
      return 'No matching farmer records found. Try asking about specific crops, states, or farming conditions.';
    }

    const lines = results.slice(0, 3).map((r, i) => {
      const p = r.payload || {};
      return (
        `${i + 1}. ${p.farmerName} from ${p.state} — ` +
        `grows ${p.crop} (${p.season}), ${p.soilType} soil, ` +
        `${p.irrigationType} irrigation, ${p.areaAcres} acres`
      );
    });

    return `Found ${results.length} relevant records:\n\n${lines.join('\n')}\n\n` +
      `(Install Gemma 2B model for AI-generated responses)`;
  }

  dispose() {
    this.vectorSearch.dispose();
    this.embedder.dispose();
    this.llama?.release();
    this.llama = null;
  }
}
