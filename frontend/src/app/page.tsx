"use client"

export default function Home() {
  return (
    <div className="w-full max-w-5xl bg-white flex flex-col rounded-2xl shadow-2xl border border-gray-100 overflow-hidden min-h-[70vh] p-8">
      <div className="flex flex-1 flex-col items-center justify-center text-center p-12">
        <h1 className="text-3xl font-bold mb-4">Welcome to the CZSU Data Explorer</h1>
        <p className="text-xl text-gray-700 max-w-2xl">
          This application contains data from the Czech Statistical Office (CZSU).<br />
          You can chat with the data using natural language, explore datasets, and filter or search tables interactively.
        </p>
      </div>
    </div>
  );
}