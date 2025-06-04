'use client';
import Link from 'next/link';
import { usePathname } from 'next/navigation';
import AuthButton from './AuthButton';

const menuItems = [
  { label: 'HOME', href: '/' },
  { label: 'CHAT', href: '/chat' },
  { label: 'CATALOG', href: '/catalog' },
  { label: 'DATA', href: '/data' },
  { label: 'CONTACTS', href: '/contacts' },
];

const Header = () => {
  const pathname = usePathname();
  return (
    <header className="relative flex items-center justify-between px-10 py-6 bg-white shadow-md z-20">
      <div className="absolute inset-0 bg-[url('/api/placeholder/100/100')] opacity-5 mix-blend-overlay pointer-events-none"></div>
      <div className="absolute bottom-0 left-0 right-0 h-px bg-gradient-to-r from-transparent via-white/20 to-transparent"></div>

      <div className="flex items-center relative">
        <span className="font-extrabold text-[#181C3A] text-2xl tracking-tight" style={{fontFamily: 'var(--font-inter)'}}>CZSU - Multi-Agent Text-to-SQL</span>
      </div>

      <nav className="flex items-center space-x-6">
        {menuItems.map(item => {
          const isActive = pathname === item.href;
          return (
            <Link
              key={item.href}
              className={
                `text-base px-3 py-2 font-semibold rounded-lg transition-all duration-200 cursor-pointer ` +
                (isActive
                  ? 'text-[#181C3A] font-extrabold bg-gray-100 shadow-sm '
                  : 'text-[#181C3A]/80 hover:text-[#181C3A] hover:bg-gray-50 ')
              }
              style={{fontFamily: 'var(--font-inter)'}} 
              href={item.href}
            >
              {item.label.charAt(0) + item.label.slice(1).toLowerCase()}
            </Link>
          );
        })}
        <div className="ml-6"><AuthButton compact={true} /></div>
      </nav>
    </header>
  );
}

export default Header