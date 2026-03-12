import React, { useState, useRef, useEffect } from 'react';
import {
  View,
  Text,
  TextInput,
  FlatList,
  TouchableOpacity,
  StyleSheet,
  KeyboardAvoidingView,
  Platform,
  ActivityIndicator,
} from 'react-native';
import type { NativeStackScreenProps } from '@react-navigation/native-stack';
import type { RootStackParamList } from '../AppNavigator';
import { ChatService, type ChatMessage } from '../services/ChatService';

type Props = NativeStackScreenProps<RootStackParamList, 'Chat'>;

export default function ChatScreen({}: Props) {
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [input, setInput] = useState('');
  const [loading, setLoading] = useState(false);
  const [status, setStatus] = useState('Initializing...');
  const [ready, setReady] = useState(false);
  const flatListRef = useRef<FlatList>(null);
  const chatService = useRef<ChatService | null>(null);

  useEffect(() => {
    const init = async () => {
      try {
        chatService.current = new ChatService();
        await chatService.current.initialize((msg) => setStatus(msg));
        setReady(true);
        setStatus('');
        setMessages([
          {
            role: 'assistant',
            content:
              'Hello! I can help you find information about farmers across India. ' +
              'Ask me about crops, soil types, weather conditions, or any farming topic.',
          },
        ]);
      } catch (err: any) {
        setStatus(`Error: ${err.message}`);
      }
    };
    init();
    return () => {
      chatService.current?.dispose();
    };
  }, []);

  const sendMessage = async () => {
    if (!input.trim() || !chatService.current || loading) return;

    const userMsg: ChatMessage = { role: 'user', content: input.trim() };
    setMessages((prev) => [...prev, userMsg]);
    setInput('');
    setLoading(true);

    try {
      const response = await chatService.current.chat(userMsg.content);
      setMessages((prev) => [
        ...prev,
        { role: 'assistant', content: response },
      ]);
    } catch (err: any) {
      setMessages((prev) => [
        ...prev,
        { role: 'assistant', content: `Error: ${err.message}` },
      ]);
    } finally {
      setLoading(false);
    }
  };

  const renderMessage = ({ item }: { item: ChatMessage }) => (
    <View
      style={[
        styles.messageBubble,
        item.role === 'user' ? styles.userBubble : styles.assistantBubble,
      ]}
    >
      <Text
        style={[
          styles.messageText,
          item.role === 'user' ? styles.userText : styles.assistantText,
        ]}
      >
        {item.content}
      </Text>
    </View>
  );

  return (
    <KeyboardAvoidingView
      style={styles.container}
      behavior={Platform.OS === 'ios' ? 'padding' : 'height'}
      keyboardVerticalOffset={90}
    >
      {status ? (
        <View style={styles.statusBar}>
          <ActivityIndicator size="small" color="#e94560" />
          <Text style={styles.statusText}>{status}</Text>
        </View>
      ) : null}

      <FlatList
        ref={flatListRef}
        data={messages}
        renderItem={renderMessage}
        keyExtractor={(_, i) => String(i)}
        contentContainerStyle={styles.messageList}
        onContentSizeChange={() =>
          flatListRef.current?.scrollToEnd({ animated: true })
        }
      />

      <View style={styles.inputRow}>
        <TextInput
          style={styles.input}
          value={input}
          onChangeText={setInput}
          placeholder="Ask about farmers..."
          placeholderTextColor="#555"
          editable={ready && !loading}
          onSubmitEditing={sendMessage}
          returnKeyType="send"
        />
        <TouchableOpacity
          style={[styles.sendBtn, (!ready || loading) && styles.disabled]}
          onPress={sendMessage}
          disabled={!ready || loading}
        >
          {loading ? (
            <ActivityIndicator size="small" color="#fff" />
          ) : (
            <Text style={styles.sendText}>Send</Text>
          )}
        </TouchableOpacity>
      </View>
    </KeyboardAvoidingView>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#0f0f23',
  },
  statusBar: {
    flexDirection: 'row',
    alignItems: 'center',
    padding: 12,
    backgroundColor: '#1a1a2e',
    gap: 8,
  },
  statusText: {
    color: '#8888aa',
    fontSize: 13,
  },
  messageList: {
    padding: 16,
    paddingBottom: 8,
  },
  messageBubble: {
    maxWidth: '80%',
    padding: 12,
    borderRadius: 16,
    marginBottom: 8,
  },
  userBubble: {
    backgroundColor: '#e94560',
    alignSelf: 'flex-end',
    borderBottomRightRadius: 4,
  },
  assistantBubble: {
    backgroundColor: '#1a1a2e',
    alignSelf: 'flex-start',
    borderBottomLeftRadius: 4,
    borderWidth: 1,
    borderColor: '#2a2a4e',
  },
  messageText: {
    fontSize: 15,
    lineHeight: 21,
  },
  userText: {
    color: '#ffffff',
  },
  assistantText: {
    color: '#ccccdd',
  },
  inputRow: {
    flexDirection: 'row',
    padding: 12,
    borderTopWidth: 1,
    borderTopColor: '#2a2a4e',
    backgroundColor: '#1a1a2e',
    gap: 8,
  },
  input: {
    flex: 1,
    backgroundColor: '#0f0f23',
    borderRadius: 20,
    paddingHorizontal: 16,
    paddingVertical: 10,
    color: '#ffffff',
    fontSize: 15,
  },
  sendBtn: {
    backgroundColor: '#e94560',
    borderRadius: 20,
    paddingHorizontal: 20,
    justifyContent: 'center',
  },
  disabled: {
    opacity: 0.4,
  },
  sendText: {
    color: '#ffffff',
    fontWeight: '600',
  },
});
