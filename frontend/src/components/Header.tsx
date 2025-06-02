'use client';
import Link from 'next/link';
import { usePathname } from 'next/navigation';

const menuItems = [
  { label: 'HOME', href: '/' },
  { label: 'CHAT', href: '/chat' },
  { label: 'DATASETS', href: '/datasets' },
  { label: 'DATA', href: '/data' },
  { label: 'CONTACTS', href: '/contacts' },
];

const Header = () => {
  const pathname = usePathname();
  return (
    <header className="relative flex items-center justify-between px-8 py-5 bg-gradient-to-r from-[#4A3F71] to-[#5E507F] z-10">
      <div className="absolute inset-0 bg-[url('/api/placeholder/100/100')] opacity-5 mix-blend-overlay pointer-events-none"></div>
      <div className="absolute bottom-0 left-0 right-0 h-px bg-gradient-to-r from-transparent via-white/20 to-transparent"></div>

      <div className="flex items-center relative">
        <div className="absolute -left-3 top-1/2 transform -translate-y-1/2 w-1.5 h-6 bg-teal-400 rounded-full opacity-80"></div>
        <span className="font-bold text-white text-xl tracking-tight">CZSU - Multi-Agent Text-to-SQL</span>
      </div>

      <div className="flex items-center space-x-1">
        {menuItems.map(item => {
          const isActive = pathname === item.href;
          return (
            <Link
              key={item.href}
              className={
                `text-xs px-4 py-2 font-medium rounded-lg transition-all duration-200 cursor-pointer ` +
                (isActive
                  ? 'text-white bg-white/10 '
                  : 'text-white/80 hover:text-white hover:bg-white/10 ')
              }
              href={item.href}
            >
              {item.label}
            </Link>
          );
        })}
      </div>
    </header>
  );
}

export default Header