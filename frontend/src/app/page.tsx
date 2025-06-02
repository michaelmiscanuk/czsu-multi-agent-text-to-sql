"use client"

import Header from '@/components/Header';
import InputBar from '@/components/InputBar';
import MessageArea from '@/components/MessageArea';
import React, { useState, useEffect } from 'react';
import DatasetsTable from '../components/DatasetsTable';
import DataTableView from '../components/DataTableView';

interface Message {
  id: number;
  content: string;
  isUser: boolean;
  type: string;
  isLoading?: boolean;
}

// Simple modal component
const Modal: React.FC<{ open: boolean; onClose: () => void; children: React.ReactNode }> = ({ open, onClose, children }) => {
  useEffect(() => {
    if (!open) return;
    const handleKeyDown = (e: KeyboardEvent) => {
      if (e.key === 'Escape') onClose();
    };
    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, [open, onClose]);
  if (!open) return null;
  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black bg-opacity-40">
      <div className="bg-white rounded-lg shadow-lg max-w-2xl w-full p-6 relative">
        <button
          className="absolute top-2 right-2 text-gray-400 hover:text-gray-700 text-2xl font-bold"
          onClick={onClose}
          title="Close"
        >
          ×
        </button>
        {children}
      </div>
    </div>
  );
};

const Home = () => {
  const [selectedMenu, setSelectedMenu] = useState('chat');
  const [messages, setMessages] = useState<Message[]>([
    {
      id: 1,
      content: 'Hi there, how can I help you?',
      isUser: false,
      type: 'message'
    }
  ]);
  const [currentMessage, setCurrentMessage] = useState("");
  const [isLoading, setIsLoading] = useState(false);

  // DataTableView state lifted up
  const [selectedTable, setSelectedTable] = useState<string | null>(null);
  const [columns, setColumns] = useState<string[]>([]);
  const [rows, setRows] = useState<any[][]>([]);
  const [selectedColumn, setSelectedColumn] = useState<string | null>(null);
  const [columnFilters, setColumnFilters] = useState<{ [col: string]: string }>({});
  const [pendingTableSearch, setPendingTableSearch] = useState<string | null>(null);
  const [lastSelectionCode, setLastSelectionCode] = useState<string | null>(null);
  const [lastQueriesAndResults, setLastQueriesAndResults] = useState<[string, string][]>([]);
  const [showSQLModal, setShowSQLModal] = useState(false);

  const handleSubmit = async (e: React.FormEvent<HTMLFormElement>) => {
    e.preventDefault();
    setLastQueriesAndResults([]);
    if (currentMessage.trim()) {
      const newMessageId = messages.length > 0 ? Math.max(...messages.map((msg: Message) => msg.id)) + 1 : 1;
      setMessages((prev: Message[]) => [
        ...prev,
        {
          id: newMessageId,
          content: currentMessage,
          isUser: true,
          type: 'message'
        }
      ]);
      const userInput = currentMessage;
      setCurrentMessage("");
      setIsLoading(true);
      try {
        // Add loading placeholder for AI response
        const aiResponseId = newMessageId + 1;
        setMessages((prev: Message[]) => [
          ...prev,
          {
            id: aiResponseId,
            content: "",
            isUser: false,
            type: 'message',
            isLoading: true
          }
        ]);
        // Call your FastAPI backend
        const API_URL = process.env.NODE_ENV === 'development'
          ? 'http://localhost:8000/analyze'
          : '/analyze';
        const response = await fetch(API_URL, {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json'
          },
          body: JSON.stringify({ prompt: userInput })
        });
        if (!response.ok) {
          throw new Error('Server error');
        }
        const data = await response.json();
        console.log('API queries_and_results:', data.queries_and_results);
        setMessages((prev: Message[]) =>
          prev.map((msg: Message) =>
            msg.id === aiResponseId
              ? { ...msg, content: data.result || JSON.stringify(data), isLoading: false }
              : msg
          )
        );
        setLastSelectionCode(data.selection_with_possible_answer || null);
        if (Array.isArray(data.queries_and_results)) {
          setLastQueriesAndResults(data.queries_and_results);
        } else {
          setLastQueriesAndResults([]);
        }
      } catch (error) {
        setMessages((prev: Message[]) =>
          prev.map((msg: Message) =>
            msg.isLoading
              ? { ...msg, content: "Sorry, there was an error processing your request.", isLoading: false }
              : msg
          )
        );
      } finally {
        setIsLoading(false);
      }
    }
  };

  // Handler to integrate Datasets and Data views
  const handleDatasetRowClick = (selection_code: string) => {
    setPendingTableSearch(selection_code);
    setSelectedMenu('data');
  };

  // Handler for clicking the dataset link below the answer
  const handleSelectionCodeClick = () => {
    if (lastSelectionCode) {
      setPendingTableSearch(lastSelectionCode);
      setSelectedMenu('data');
    }
  };

  // Handler for SQL button
  const handleSQLButtonClick = () => {
    setShowSQLModal(true);
  };

  const handleCloseSQLModal = () => {
    setShowSQLModal(false);
  };

  return (
    <div className="min-h-screen w-full bg-gradient-to-br from-gray-100 via-blue-50 to-purple-100 flex flex-col">
      <Header onMenuClick={setSelectedMenu} selectedMenu={selectedMenu} />
      <main className="flex justify-center flex-1 py-8 px-2">
        <div className="w-full max-w-5xl bg-white flex flex-col rounded-2xl shadow-2xl border border-gray-100 overflow-hidden min-h-[70vh] p-8">
          {selectedMenu === 'chat' && <>
            <MessageArea messages={messages} />
            {/* Show the dataset link and SQL button below the latest AI answer if available */}
            {lastSelectionCode &&
              <div className="mt-2 text-sm flex items-center space-x-4">
                <div>
                  <span>Dataset used: </span>
                  <button
                    className="text-blue-600 underline hover:text-blue-800 cursor-pointer font-mono"
                    onClick={handleSelectionCodeClick}
                  >
                    {lastSelectionCode}
                  </button>
                </div>
                <button
                  className="px-3 py-1 bg-gray-200 hover:bg-gray-300 rounded text-xs font-semibold text-gray-700 border border-gray-300"
                  onClick={handleSQLButtonClick}
                >
                  SQL
                </button>
              </div>
            }
            {/* SQL Modal */}
            <Modal open={showSQLModal} onClose={handleCloseSQLModal}>
              <h2 className="text-lg font-bold mb-4">SQL Commands & Results</h2>
              <div className="max-h-[60vh] overflow-y-auto pr-2">
                {(() => {
                  // Deduplicate by SQL string
                  const uniqueQueriesAndResults = Array.from(
                    new Map(lastQueriesAndResults.map(([q, r]) => [q, [q, r]])).values()
                  );
                  if (uniqueQueriesAndResults.length === 0) {
                    return <div className="text-gray-500">No SQL commands available.</div>;
                  }
                  return (
                    <div className="space-y-6">
                      {uniqueQueriesAndResults.map(([sql, result], idx) => (
                        <div key={idx} className="bg-gray-50 rounded border border-gray-200 p-0">
                          <div className="bg-gray-100 px-4 py-2 rounded-t text-xs font-semibold text-gray-700 border-b border-gray-200">SQL Command {idx + 1}</div>
                          <div className="p-3 font-mono text-xs whitespace-pre-line text-gray-900">
                            {sql.split('\n').map((line, i) => (
                              <React.Fragment key={i}>
                                {line}
                                {i !== sql.split('\n').length - 1 && <br />}
                              </React.Fragment>
                            ))}
                          </div>
                          <div className="bg-gray-100 px-4 py-2 text-xs font-semibold text-gray-700 border-t border-gray-200">Result</div>
                          <div className="p-3 font-mono text-xs whitespace-pre-line text-gray-800">
                            {typeof result === 'string' ? result : JSON.stringify(result, null, 2)}
                          </div>
                        </div>
                      ))}
                    </div>
                  );
                })()}
              </div>
            </Modal>
            <InputBar currentMessage={currentMessage} setCurrentMessage={setCurrentMessage} onSubmit={handleSubmit} isLoading={isLoading} />
          </>}
          {selectedMenu === 'datasets' && <DatasetsTable onRowClick={handleDatasetRowClick} />}
          {selectedMenu === 'data' && (
            <DataTableView
              selectedTable={selectedTable}
              setSelectedTable={setSelectedTable}
              columns={columns}
              setColumns={setColumns}
              rows={rows}
              setRows={setRows}
              selectedColumn={selectedColumn}
              setSelectedColumn={setSelectedColumn}
              columnFilters={columnFilters}
              setColumnFilters={setColumnFilters}
              pendingTableSearch={pendingTableSearch}
              setPendingTableSearch={setPendingTableSearch}
            />
          )}
          {selectedMenu === 'home' && (
            <div className="flex flex-1 flex-col items-center justify-center text-center p-12">
              <h1 className="text-3xl font-bold mb-4">Welcome to the CZSU Data Explorer</h1>
              <p className="text-xl text-gray-700 max-w-2xl">
                This application contains data from the Czech Statistical Office (CZSU).<br />
                You can chat with the data using natural language, explore datasets, and filter or search tables interactively.
              </p>
            </div>
          )}
          {selectedMenu === 'contacts' && (
            <div className="flex flex-1 flex-col items-center justify-center text-center p-12">
              <h1 className="text-2xl font-bold mb-4">Contact</h1>
              <div className="text-lg text-gray-700 max-w-xl space-y-4">
                <div><span className="font-semibold">Name:</span> Michael Miscanuk</div>
                <div>
                  <span className="font-semibold">Email:</span> <a href="mailto:michael.miscanuk@gmail.com" className="text-blue-600 underline">michael.miscanuk@gmail.com</a>
                </div>
                <div>
                  <span className="font-semibold">LinkedIn:</span> <a href="https://www.linkedin.com/in/michael-miscanuk-b9503b77/" target="_blank" rel="noopener noreferrer" className="text-blue-600 underline">michael-miscanuk-b9503b77</a>
                </div>
                <div>
                  <span className="font-semibold">GitHub:</span> <a href="https://github.com/michaelmiscanuk" target="_blank" rel="noopener noreferrer" className="text-blue-600 underline">github.com/michaelmiscanuk</a>
                </div>
                <div>
                  <span className="font-semibold">About me:</span>
                  <span className="ml-1">I'm passionate about Data Engineering, Data Science, and AI Engineering. I enjoy building intelligent systems and working with data to solve real-world problems.</span>
                </div>
              </div>
            </div>
          )}
        </div>
      </main>
      <footer className="w-full text-center text-gray-400 text-sm py-4 mt-4">
        &copy; {new Date().getFullYear()} Michael Miscanuk. Data from the Czech Statistical Office (CZSU).
      </footer>
    </div>
  );
};

export default Home;