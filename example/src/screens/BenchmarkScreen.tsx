import React, { useState, useRef } from 'react';
import {
  View,
  Text,
  TouchableOpacity,
  ScrollView,
  StyleSheet,
  ActivityIndicator,
  Platform,
} from 'react-native';
import type { NativeStackScreenProps } from '@react-navigation/native-stack';
import type { RootStackParamList } from '../AppNavigator';
import { BenchmarkRunner, PHASES, type PhaseConfig } from '../services/BenchmarkRunner';

type Props = NativeStackScreenProps<RootStackParamList, 'Benchmark'>;

export default function BenchmarkScreen({ navigation }: Props) {
  const [phases, setPhases] = useState<PhaseConfig[]>(
    PHASES.map((p) => ({ ...p })),
  );
  const [running, setRunning] = useState(false);
  const [log, setLog] = useState<string[]>([]);
  const [progress, setProgress] = useState(0);
  const [progressLabel, setProgressLabel] = useState('');
  const runner = useRef<BenchmarkRunner | null>(null);

  const togglePhase = (idx: number) => {
    setPhases((prev) =>
      prev.map((p, i) => (i === idx ? { ...p, enabled: !p.enabled } : p)),
    );
  };

  const startBenchmark = async () => {
    const enabled = phases.filter((p) => p.enabled);
    if (enabled.length === 0) return;

    setRunning(true);
    setLog([]);
    setProgress(0);
    setProgressLabel('');

    try {
      runner.current = new BenchmarkRunner();
      const results = await runner.current.run(
        phases,
        (msg) => setLog((prev) => [...prev, msg]),
        (frac, label) => {
          setProgress(Math.min(frac, 1));
          setProgressLabel(label);
        },
      );
      console.log('BENCHMARK_RESULTS_JSON:' + JSON.stringify(results));
      navigation.navigate('Results', { results: JSON.stringify(results) });
    } catch (err: any) {
      setLog((prev) => [...prev, `ERROR: ${err.message}`]);
    } finally {
      setRunning(false);
    }
  };

  return (
    <View style={styles.container}>
      <Text style={styles.sectionTitle}>Benchmark Phases</Text>

      {phases.map((phase, idx) => (
        <TouchableOpacity
          key={phase.id}
          style={styles.engineRow}
          onPress={() => togglePhase(idx)}
          disabled={running}
        >
          <View
            style={[styles.checkbox, phase.enabled && styles.checkboxActive]}
          >
            {phase.enabled && <Text style={styles.checkmark}>✓</Text>}
          </View>
          <Text style={styles.engineName}>
            P{phase.id}: {phase.name}
          </Text>
        </TouchableOpacity>
      ))}

      <Text style={styles.sectionTitle}>Parameters</Text>
      <View style={styles.settingRow}>
        <Text style={styles.settingLabel}>Queries per sweep</Text>
        <Text style={styles.settingValue}>100</Text>
      </View>
      <View style={styles.settingRow}>
        <Text style={styles.settingLabel}>Top-K</Text>
        <Text style={styles.settingValue}>10</Text>
      </View>
      <View style={styles.settingRow}>
        <Text style={styles.settingLabel}>Warmup</Text>
        <Text style={styles.settingValue}>10</Text>
      </View>
      <View style={styles.settingRow}>
        <Text style={styles.settingLabel}>EF sweep</Text>
        <Text style={styles.settingValue}>16…500</Text>
      </View>

      {running && (
        <View style={styles.progressContainer}>
          <View style={styles.progressTrack}>
            <View style={[styles.progressFill, { width: `${Math.round(progress * 100)}%` }]} />
          </View>
          <Text style={styles.progressText}>
            {progressLabel} ({Math.round(progress * 100)}%)
          </Text>
        </View>
      )}

      {log.length > 0 && (
        <ScrollView style={styles.logBox}>
          {log.map((line, i) => (
            <Text key={i} style={styles.logLine}>
              {line}
            </Text>
          ))}
        </ScrollView>
      )}

      <TouchableOpacity
        style={[styles.runBtn, running && styles.disabled]}
        onPress={startBenchmark}
        disabled={running}
      >
        {running ? (
          <ActivityIndicator size="small" color="#fff" />
        ) : (
          <Text style={styles.runBtnText}>Run Benchmark</Text>
        )}
      </TouchableOpacity>
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#0f0f23',
    padding: 24,
  },
  sectionTitle: {
    fontSize: 14,
    fontWeight: '600',
    color: '#e94560',
    textTransform: 'uppercase',
    letterSpacing: 1,
    marginBottom: 12,
    marginTop: 8,
  },
  engineRow: {
    flexDirection: 'row',
    alignItems: 'center',
    padding: 12,
    backgroundColor: '#1a1a2e',
    borderRadius: 12,
    marginBottom: 8,
    gap: 12,
  },
  checkbox: {
    width: 24,
    height: 24,
    borderRadius: 6,
    borderWidth: 2,
    borderColor: '#555',
    justifyContent: 'center',
    alignItems: 'center',
  },
  checkboxActive: {
    borderColor: '#e94560',
    backgroundColor: '#e94560',
  },
  checkmark: {
    color: '#fff',
    fontWeight: 'bold',
    fontSize: 14,
  },
  engineName: {
    color: '#ffffff',
    fontSize: 16,
    fontWeight: '500',
  },
  settingRow: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    paddingVertical: 8,
    paddingHorizontal: 4,
    borderBottomWidth: 1,
    borderBottomColor: '#1a1a2e',
  },
  settingLabel: {
    color: '#8888aa',
    fontSize: 14,
  },
  settingValue: {
    color: '#ffffff',
    fontSize: 14,
    fontWeight: '500',
  },
  logBox: {
    backgroundColor: '#1a1a2e',
    borderRadius: 12,
    padding: 12,
    maxHeight: 160,
    marginTop: 16,
  },
  logLine: {
    color: '#66bb6a',
    fontSize: 12,
    fontFamily: Platform.select({ ios: 'Menlo', android: 'monospace' }),
    lineHeight: 18,
  },
  progressContainer: {
    marginTop: 16,
    marginBottom: 4,
  },
  progressTrack: {
    height: 8,
    backgroundColor: '#1a1a2e',
    borderRadius: 4,
    overflow: 'hidden',
  },
  progressFill: {
    height: '100%',
    backgroundColor: '#e94560',
    borderRadius: 4,
  },
  progressText: {
    color: '#8888aa',
    fontSize: 12,
    marginTop: 4,
    fontFamily: Platform.select({ ios: 'Menlo', android: 'monospace' }),
  },
  runBtn: {
    backgroundColor: '#e94560',
    borderRadius: 16,
    padding: 16,
    alignItems: 'center',
    marginTop: 'auto',
  },
  disabled: {
    opacity: 0.5,
  },
  runBtnText: {
    color: '#ffffff',
    fontWeight: 'bold',
    fontSize: 16,
  },
});
