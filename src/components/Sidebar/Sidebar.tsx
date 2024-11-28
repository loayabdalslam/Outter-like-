import React from 'react';
import { Home, Bell, Search, Users, Hash, MessageCircle, Folder } from 'lucide-react';

export function Sidebar() {
  return (
    <div className="fixed left-0 top-0 h-screen w-60 bg-white border-r">
      <div className="p-4">
        <div className="flex items-center space-x-2 mb-6">
          <img src="/logo.svg" alt="Logo" className="h-8 w-8" />
          <span className="font-semibold text-xl">OtterPilot</span>
        </div>
        
        <nav className="space-y-1">
          <SidebarItem icon={Home} label="Home" active />
          <SidebarItem icon={Bell} label="Notifications" />
          <SidebarItem icon={Search} label="Search" />
          <SidebarItem icon={Users} label="Team" />
          <SidebarItem icon={Hash} label="Channels" />
          <SidebarItem icon={MessageCircle} label="Direct Messages" />
          <SidebarItem icon={Folder} label="Files" />
        </nav>
      </div>
    </div>
  );
}

interface SidebarItemProps {
  icon: React.ElementType;
  label: string;
  active?: boolean;
}

function SidebarItem({ icon: Icon, label, active }: SidebarItemProps) {
  return (
    <button 
      className={`
        flex items-center space-x-2 w-full px-3 py-2 rounded-lg
        ${active 
          ? 'bg-blue-50 text-blue-600' 
          : 'text-gray-700 hover:bg-gray-100'
        }
      `}
    >
      <Icon className="h-5 w-5" />
      <span>{label}</span>
    </button>
  );
} 