"use client";

export default function ContactsPage() {
  return (
    <div className="w-full max-w-5xl bg-white flex flex-col rounded-2xl shadow-2xl border border-gray-100 overflow-hidden min-h-[70vh] p-8">
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
    </div>
  );
} 