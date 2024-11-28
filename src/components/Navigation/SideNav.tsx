import React from 'react';
import { MessageSquare, List, Info } from 'lucide-react';

export function SideNav() {
  return (
    <nav className="fixed left-0 top-0 bottom-0 w-16 bg-white border-r flex flex-col items-center py-4">
      <div className="w-8 h-8 bg-blue-600 rounded mb-8" />
      <button className="w-10 h-10 rounded-lg mb-2 hover:bg-gray-100 flex items-center justify-center">
        <MessageSquare className="w-5 h-5 text-gray-600" />
      </button>
      <button className="w-10 h-10 rounded-lg mb-2 hover:bg-gray-100 flex items-center justify-center">
        <List className="w-5 h-5 text-gray-600" />
      </button>
      <button className="w-10 h-10 rounded-lg mb-2 hover:bg-gray-100 flex items-center justify-center">
        <Info className="w-5 h-5 text-gray-600" />
      </button>
    </nav>
  );
}