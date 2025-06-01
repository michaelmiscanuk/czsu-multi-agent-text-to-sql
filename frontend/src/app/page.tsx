"use client"

import Header from '@/components/Header';
import InputBar from '@/components/InputBar';
import MessageArea from '@/components/MessageArea';
import React, { useState } from 'react';

interface Message {
  id: number;
  content: string;
  isUser: boolean;
  type: string;
  isLoading?: boolean;
}

const Home = () => {
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

  return (
    <div className="flex justify-center bg-gray-100 min-h-screen py-8 px-4">
      {/* Main container with refined shadow and border */}
      <div className="w-[70%] bg-white flex flex-col rounded-xl shadow-lg border border-gray-100 overflow-hidden h-[90vh]">
        <Header />
        <MessageArea messages={messages} />
        <InputBar currentMessage={currentMessage} setCurrentMessage={setCurrentMessage} onSubmit={handleSubmit} isLoading={isLoading} />
      </div>
    </div>
  );
};

export default Home;