import { useEffect, useState } from "react";
import { api } from "./api";
import { Chat } from "./components/Chat";
import { ArtifactFeed, BandBars, QualityGrid, StateCard } from "./components/BrainCards";
import { ConnectivityCard } from "./components/ConnectivityCard";
import { ConversationList } from "./components/ConversationList";
import { Header } from "./components/Header";
import { LLMSettings } from "./components/LLMSettings";
import { LogsPanel } from "./components/LogsPanel";
import { PatternModal } from "./components/PatternModal";
import { PatternTicker } from "./components/PatternTicker";
import { SystemControl } from "./components/SystemControl";
import { toast, ToastContainer } from "./components/Toast";
import { TrainingOverlay } from "./components/TrainingOverlay";
import { useChatSocket } from "./hooks/useChatSocket";
import { useLogsCapture } from "./hooks/useLogsCapture";
import { useLiveSocket } from "./hooks/useLiveSocket";
import { useTrainSocket } from "./hooks/useTrainSocket";
import type { Info, PatternExplain } from "./types";

export default function App() {
  const logs = useLogsCapture();
  const { snapshot, connected } = useLiveSocket();
  const chat = useChatSocket(logs);
  const train = useTrainSocket();

  const [info, setInfo] = useState<Info | null>(null);
  const [training, setTraining] = useState(false);
  const [showSettings, setShowSettings] = useState(false);
  const [showSystem, setShowSystem] = useState(false);
  const [showLogs, setShowLogs] = useState(false);
  const [showHistory, setShowHistory] = useState(false);
  const [explain, setExplain] = useState<{ name: string; data: PatternExplain | null } | null>(null);

  useEffect(() => {
    api.info().then(setInfo).catch(() => {});
  }, []);

  const openExplain = async (name: string) => {
    setExplain({ name, data: null });
    try {
      const data = await api.explain(name);
      setExplain({ name, data });
    } catch {
      setExplain({ name, data: { error: "failed to load" } });
    }
  };

  const forget = async (name: string) => {
    try {
      await api.forget(name);
      toast.success(`Pattern "${name}" deleted`);
      // The live /ws/live snapshot reflects the removal on its next push.
    } catch (err) {
      toast.error("Failed to delete pattern");
    }
  };

  return (
    <div className="app">
      <Header 
        info={info} 
        connected={connected} 
        onSettings={() => setShowSettings(true)}
        onSystem={() => setShowSystem(true)}
        onLogs={() => setShowLogs(true)}
        onHistory={() => setShowHistory(true)}
      />
      <main className="layout">
        <Chat
          messages={chat.messages}
          busy={chat.busy}
          connected={chat.connected}
          onSend={chat.send}
          onReset={chat.reset}
        />
        <aside className="brain">
          <StateCard state={snapshot?.state} />
          <BandBars bands={snapshot?.bands} />
          <QualityGrid quality={snapshot?.quality} />
          <ConnectivityCard conn={snapshot?.connectivity || {}} />
          <PatternTicker
            patterns={snapshot?.patterns}
            onTrain={() => setTraining(true)}
            onExplain={openExplain}
            onForget={forget}
          />
          <ArtifactFeed artifacts={snapshot?.artifacts} />
        </aside>
      </main>

      {training && <TrainingOverlay train={train} onClose={() => setTraining(false)} />}
      {explain && (
        <PatternModal name={explain.name} data={explain.data} onClose={() => setExplain(null)} />
      )}
      {showSettings && <LLMSettings info={info} onClose={() => setShowSettings(false)} />}
      {showSystem && <SystemControl onClose={() => setShowSystem(false)} />}
      {showHistory && (
        <ConversationList
          currentId={chat.conversationId}
          onSelect={chat.loadConversation}
          onNew={chat.newConversation}
          onClose={() => setShowHistory(false)}
        />
      )}
      {showLogs && (
        <LogsPanel
          logs={logs.logs}
          enabled={logs.enabled}
          filter={logs.filter}
          onToggleEnabled={logs.setEnabled}
          onFilterChange={logs.setFilter}
          onToggleExpanded={logs.toggleExpanded}
          onClear={logs.clearLogs}
          onClose={() => setShowLogs(false)}
        />
      )}
      <ToastContainer />
    </div>
  );
}
