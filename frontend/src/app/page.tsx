export default function Home() {
  return (
    <div className="w-full max-w-5xl bg-white flex flex-col rounded-2xl shadow-2xl border border-gray-100 overflow-hidden min-h-[70vh] p-8">
      <div className="flex flex-1 flex-col items-center justify-center text-center p-12">
        <h1 className="text-3xl font-bold mb-4">Welcome to the CZSU Data Explorer and Chatbot</h1>
        <p className="text-xl text-gray-700 max-w-2xl">
          This application contains data from the Czech Statistical Office (CZSU).<br />
          You can chat with the data using natural language, explore the catalog, and filter or search tables interactively.
        </p>
        <div className="flex gap-8 mt-8">
          <a
            href="https://csu.gov.cz/zakladni-informace-pro-pouziti-api-datastatu"
            target="_blank"
            rel="noopener noreferrer"
            className="text-3xl font-bold text-blue-600 hover:text-blue-800 underline"
          >
            API
          </a>
          <a
            href="https://csu.gov.cz/docs/107508/db2ef2f9-5b1f-82c2-5ebf-621acf94791d/32019824.pdf?version=1.2"
            target="_blank"
            rel="noopener noreferrer"
            className="text-3xl font-bold text-blue-600 hover:text-blue-800 underline"
          >
            PDF
          </a>
        </div>
      </div>
    </div>
  );
}