import { useEffect, useState } from "react";
import { api } from "./api";
import { Chat } from "./components/Chat";
import { ArtifactFeed, BandBars, QualityGrid, StateCard } from "./components/BrainCards";
import { CascadeMonitor } from "./components/CascadeMonitor";
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
import { TrendsCard } from "./components/TrendsCard";
import { useChatSocket } from "./hooks/useChatSocket";
import { useLogsCapture } from "./hooks/useLogsCapture";
import { useLiveSocket } from "./hooks/useLiveSocket";
import { useTrainSocket } from "./hooks/useTrainSocket";
import { useTrendHistory } from "./hooks/useTrendHistory";
import type { Info, PatternExplain } from "./types";

export default function App() {
  const logs = useLogsCapture();
  const { snapshot, connected } = useLiveSocket();
  const chat = useChatSocket(logs);
  const train = useTrainSocket();
  const trends = useTrendHistory(snapshot);

  const [info, setInfo] = useState<Info | null>(null);
  const [training, setTraining] = useState(false);
  const [showSettings, setShowSettings] = useState(false);
  const [showSystem, setShowSystem] = useState(false);
  const [showLogs, setShowLogs] = useState(false);
  const [sidebarOpen, setSidebarOpen] = useState(true);
  const [explain, setExplain] = useState<{ name: string; data: PatternExplain | null } | null>(null);

  const refreshInfo = async () => {
    try {
      const data = await api.info();
      if (data && typeof data === 'object') {
        setInfo(data);
      }
    } catch (err) {
      console.warn('[App] Failed to fetch /api/info:', err);
    }
  };

  useEffect(() => {
    let retryCount = 0;
    const maxRetries = 5;
    const retryDelay = 1000;

    const fetchInfo = async () => {
      try {
        const data = await api.info();
        // Validate that we got valid data
        if (data && typeof data === 'object') {
          setInfo(data);
        } else {
          throw new Error('Invalid info response');
        }
      } catch (err) {
        console.warn('[App] Failed to fetch /api/info:', err);
        retryCount++;
        if (retryCount < maxRetries) {
          setTimeout(fetchInfo, retryDelay * retryCount);
        } else {
          console.error('[App] Failed to fetch /api/info after', maxRetries, 'attempts');
        }
      }
    };

    fetchInfo();
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
    <div className={`app ${sidebarOpen ? "sidebar-open" : ""}`}>
      <Header 
        info={info} 
        connected={connected} 
        onSettings={() => setShowSettings(true)}
        onSystem={() => setShowSystem(true)}
        onLogs={() => setShowLogs(true)}
        onHistory={() => setSidebarOpen(!sidebarOpen)}
        sidebarOpen={sidebarOpen}
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
          <TrendsCard points={trends.points} onClear={trends.clear} />
          <StateCard state={snapshot?.state} />
          <BandBars bands={snapshot?.bands} />
          <QualityGrid quality={snapshot?.quality} />
          <ConnectivityCard conn={snapshot?.connectivity || {}} />
          <CascadeMonitor
            cascade={snapshot?.cascade}
            state={snapshot?.state}
            events={snapshot?.events}
          />
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
      {showSettings && <LLMSettings info={info} onClose={() => setShowSettings(false)} onSaved={refreshInfo} />}
      {showSystem && <SystemControl onClose={() => setShowSystem(false)} />}
      <ConversationList
        currentId={chat.conversationId}
        onSelect={chat.loadConversation}
        onNew={chat.newConversation}
        isOpen={sidebarOpen}
        onToggle={() => setSidebarOpen(!sidebarOpen)}
      />
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
