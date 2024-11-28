import React from 'react';
import { Wand2 } from 'lucide-react';

export function AIChatSidebar() {
  return (
    <div className="fixed top-0 right-0 w-80 h-screen border-l bg-white">
      <div className="flex border-b">
        <button className="flex-1 px-4 py-3 text-blue-600 border-b-2 border-blue-600">
          AI Chat
        </button>
        <button className="flex-1 px-4 py-3 text-gray-500 hover:bg-gray-50">
          Outline
        </button>
        <button className="flex-1 px-4 py-3 text-gray-500 hover:bg-gray-50">
          Comments
        </button>
      </div>

      <div className="p-4">
        <h2 className="text-lg font-semibold mb-4">
          Ask AI questions or chat with your teammates
        </h2>

        <div className="space-y-2">
          <AIChatButton text="Have I been mentioned so far in the meeting?" />
          <AIChatButton text="What decisions have been made so far?" />
          <AIChatButton text="Catch me up on the conversation." />
        </div>

        <div className="absolute bottom-4 left-4 right-4">
          <div className="relative">
            <input
              type="text"
              placeholder="Ask Otter anything about your conversations"
              className="w-full px-4 py-2 pr-10 border rounded-full bg-gray-50 focus:outline-none focus:ring-2 focus:ring-blue-500"
            />
            <Wand2 className="absolute right-3 top-2.5 h-5 w-5 text-gray-400" />
          </div>
        </div>
      </div>
    </div>
  );
}

function AIChatButton({ text }: { text: string }) {
  return (
    <button className="w-full text-left p-3 rounded-lg hover:bg-gray-50 flex items-center gap-2">
      <Wand2 className="w-4 h-4 text-blue-600" />
      <span>{text}</span>
    </button>
  );
}