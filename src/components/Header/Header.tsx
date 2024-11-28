import React from 'react';
import { Share2 } from 'lucide-react';

export function Header() {
  return (
    <header className="bg-white border-b px-4 py-2 flex items-center justify-between">
      <div className="flex items-center gap-2">
        <span className="font-semibold">General</span>
        <span className="text-gray-500">/</span>
        <span className="text-gray-500">Note</span>
      </div>
      <div className="flex items-center gap-4">
        <button className="flex items-center gap-2 px-4 py-2 text-blue-600 hover:bg-blue-50 rounded-md">
          <Share2 className="w-4 h-4" />
          Share
        </button>
      </div>
    </header>
  );
}