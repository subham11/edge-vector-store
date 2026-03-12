import React from 'react';
import {
  View,
  Text,
  TouchableOpacity,
  StyleSheet,
  StatusBar,
} from 'react-native';
import type { NativeStackScreenProps } from '@react-navigation/native-stack';
import type { RootStackParamList } from '../AppNavigator';

type Props = NativeStackScreenProps<RootStackParamList, 'Home'>;

export default function HomeScreen({ navigation }: Props) {
  return (
    <View style={styles.container}>
      <StatusBar barStyle="light-content" />
      <Text style={styles.title}>Edge Vector Store</Text>
      <Text style={styles.subtitle}>
        On-device vector search for 100K farmer records
      </Text>

      <TouchableOpacity
        style={styles.card}
        onPress={() => navigation.navigate('Chat')}
      >
        <Text style={styles.cardEmoji}>💬</Text>
        <Text style={styles.cardTitle}>Farmer Assistant</Text>
        <Text style={styles.cardDesc}>
          RAG-powered chat using Gemma 2B + vector search
        </Text>
      </TouchableOpacity>

      <TouchableOpacity
        style={styles.card}
        onPress={() => navigation.navigate('Benchmark')}
      >
        <Text style={styles.cardEmoji}>📊</Text>
        <Text style={styles.cardTitle}>Benchmark</Text>
        <Text style={styles.cardDesc}>
          Compare EdgeVectorStore vs USearch vs brute-force
        </Text>
      </TouchableOpacity>
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#0f0f23',
    padding: 24,
    paddingTop: 48,
  },
  title: {
    fontSize: 28,
    fontWeight: 'bold',
    color: '#e94560',
    marginBottom: 8,
  },
  subtitle: {
    fontSize: 14,
    color: '#8888aa',
    marginBottom: 32,
  },
  card: {
    backgroundColor: '#1a1a2e',
    borderRadius: 16,
    padding: 20,
    marginBottom: 16,
    borderWidth: 1,
    borderColor: '#2a2a4e',
  },
  cardEmoji: {
    fontSize: 32,
    marginBottom: 8,
  },
  cardTitle: {
    fontSize: 18,
    fontWeight: '600',
    color: '#ffffff',
    marginBottom: 4,
  },
  cardDesc: {
    fontSize: 13,
    color: '#8888aa',
  },
});
