import React from 'react';

interface LoadingSpinnerProps {
  size?: 'sm' | 'md' | 'lg';
  text?: string;
  className?: string;
}

export function LoadingSpinner({ size = 'md', text, className = '' }: LoadingSpinnerProps) {
  const sizeClasses = {
    sm: 'w-4 h-4',
    md: 'w-6 h-6',
    lg: 'w-8 h-8'
  };

  const textSizes = {
    sm: 'text-xs',
    md: 'text-sm',
    lg: 'text-base'
  };

  return (
    <div className={`flex flex-col items-center justify-center py-4 ${className}`}>
      <div 
        className={`${sizeClasses[size]} border-2 border-blue-500 border-t-transparent rounded-full animate-spin mb-2`}
        role="status"
        aria-label="Loading"
      />
      {text && (
        <div className={`${textSizes[size]} text-gray-500 text-center`}>
          {text}
        </div>
      )}
    </div>
  );
}

export default LoadingSpinner; 