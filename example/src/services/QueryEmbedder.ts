// QueryEmbedder — Embeds user queries using bundled ONNX MiniLM model
//
// Uses onnxruntime-react-native to run all-MiniLM-L6-v2 (~22 MB) on-device.
// Falls back to random vectors if the model isn't loaded (for testing).

import { InferenceSession, Tensor } from 'onnxruntime-react-native';
import RNFS from 'react-native-fs';

const MODEL_FILENAME = 'all-MiniLM-L6-v2.onnx';
const DIMS = 384;
const MAX_SEQ_LEN = 128;

export class QueryEmbedder {
  private session: InferenceSession | null = null;

  async initialize(
    onStatus?: (msg: string) => void,
  ): Promise<void> {
    // Look for the model in the app's document or bundle directory
    const candidates = [
      `${RNFS.DocumentDirectoryPath}/${MODEL_FILENAME}`,
      `${RNFS.MainBundlePath}/${MODEL_FILENAME}`,
      `${RNFS.MainBundlePath}/assets/${MODEL_FILENAME}`,
    ];

    let modelPath: string | null = null;
    for (const path of candidates) {
      if (await RNFS.exists(path)) {
        modelPath = path;
        break;
      }
    }

    if (!modelPath) {
      onStatus?.(`ONNX model not found — using fallback random embeddings`);
      return;
    }

    onStatus?.('Loading embedding model...');
    this.session = await InferenceSession.create(modelPath);
    onStatus?.('Embedding model ready');
  }

  async embed(text: string): Promise<Float32Array> {
    if (!this.session) {
      return this.fallbackEmbed(text);
    }

    // Simple whitespace tokenization → integer IDs
    // For production, use a proper tokenizer; this is a minimal placeholder
    const tokens = this.simpleTokenize(text);

    const inputIds = new BigInt64Array(MAX_SEQ_LEN);
    const attentionMask = new BigInt64Array(MAX_SEQ_LEN);
    const tokenTypeIds = new BigInt64Array(MAX_SEQ_LEN);

    // [CLS] = 101
    inputIds[0] = BigInt(101);
    attentionMask[0] = BigInt(1);
    for (let i = 0; i < Math.min(tokens.length, MAX_SEQ_LEN - 2); i++) {
      inputIds[i + 1] = BigInt(tokens[i]);
      attentionMask[i + 1] = BigInt(1);
    }
    // [SEP] = 102
    const sepIdx = Math.min(tokens.length + 1, MAX_SEQ_LEN - 1);
    inputIds[sepIdx] = BigInt(102);
    attentionMask[sepIdx] = BigInt(1);

    const feeds = {
      input_ids: new Tensor('int64', inputIds, [1, MAX_SEQ_LEN]),
      attention_mask: new Tensor('int64', attentionMask, [1, MAX_SEQ_LEN]),
      token_type_ids: new Tensor('int64', tokenTypeIds, [1, MAX_SEQ_LEN]),
    };

    const output = await this.session.run(feeds);
    // Mean pooling over token embeddings
    const embeddings = output['last_hidden_state']?.data as Float32Array;
    if (!embeddings) {
      return this.fallbackEmbed(text);
    }

    const result = new Float32Array(DIMS);
    const numTokens = Math.min(tokens.length + 2, MAX_SEQ_LEN);
    for (let d = 0; d < DIMS; d++) {
      let sum = 0;
      for (let t = 0; t < numTokens; t++) {
        sum += embeddings[t * DIMS + d];
      }
      result[d] = sum / numTokens;
    }

    // L2 normalize
    let norm = 0;
    for (let d = 0; d < DIMS; d++) norm += result[d] * result[d];
    norm = Math.sqrt(norm);
    if (norm > 0) {
      for (let d = 0; d < DIMS; d++) result[d] /= norm;
    }

    return result;
  }

  // Minimal hash-based tokenizer (placeholder for real WordPiece)
  private simpleTokenize(text: string): number[] {
    return text
      .toLowerCase()
      .split(/\s+/)
      .filter(Boolean)
      .map((word) => {
        let hash = 0;
        for (let i = 0; i < word.length; i++) {
          hash = ((hash << 5) - hash + word.charCodeAt(i)) | 0;
        }
        return (Math.abs(hash) % 29000) + 1000; // Map to vocab range
      });
  }

  // Deterministic fallback when ONNX model isn't available
  private fallbackEmbed(text: string): Float32Array {
    const vec = new Float32Array(DIMS);
    let seed = 0;
    for (let i = 0; i < text.length; i++) {
      seed = ((seed << 5) - seed + text.charCodeAt(i)) | 0;
    }
    for (let d = 0; d < DIMS; d++) {
      seed = (seed * 1103515245 + 12345) | 0;
      vec[d] = ((seed >> 16) & 0x7fff) / 32768.0 - 0.5;
    }
    // L2 normalize
    let norm = 0;
    for (let d = 0; d < DIMS; d++) norm += vec[d] * vec[d];
    norm = Math.sqrt(norm);
    for (let d = 0; d < DIMS; d++) vec[d] /= norm;
    return vec;
  }

  dispose() {
    this.session = null;
  }
}
