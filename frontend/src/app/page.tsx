"use client"

import Header from '@/components/Header';
import InputBar from '@/components/InputBar';
import MessageArea from '@/components/MessageArea';
import React, { useState } from 'react';
import DatasetsTable from '../components/DatasetsTable';
import DataTableView from '../components/DataTableView';

interface Message {
  id: number;
  content: string;
  isUser: boolean;
  type: string;
  isLoading?: boolean;
}

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

  const handleSubmit = async (e: React.FormEvent<HTMLFormElement>) => {
    e.preventDefault();
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
        // Show the result in the chat
        setMessages((prev: Message[]) =>
          prev.map((msg: Message) =>
            msg.id === aiResponseId
              ? { ...msg, content: data.result || JSON.stringify(data), isLoading: false }
              : msg
          )
        );
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

  return (
    <div className="min-h-screen w-full bg-gradient-to-br from-gray-100 via-blue-50 to-purple-100 flex flex-col">
      <Header onMenuClick={setSelectedMenu} selectedMenu={selectedMenu} />
      <main className="flex justify-center flex-1 py-8 px-2">
        <div className="w-full max-w-5xl bg-white flex flex-col rounded-2xl shadow-2xl border border-gray-100 overflow-hidden min-h-[70vh] p-8">
          {selectedMenu === 'chat' && <>
            <MessageArea messages={messages} />
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