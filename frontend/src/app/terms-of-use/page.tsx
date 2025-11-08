export default function TermsOfUse() {
  return (
    <div className="max-w-4xl mx-auto px-6 py-12">
      <h1 className="text-4xl font-bold text-gray-800 mb-8">Terms of Use</h1>
      
      <div className="bg-white rounded-lg shadow-md p-8 space-y-6">
        <section>
          <h2 className="text-2xl font-semibold text-gray-800 mb-4">Copyright Notice</h2>
          <p className="text-gray-700">
            © 2025 Michael Miscanuk. All rights reserved.
          </p>
        </section>

        <section>
          <h2 className="text-2xl font-semibold text-gray-800 mb-4">1. Ownership and Intellectual Property</h2>
          <p className="text-gray-700">
            All original components of this web application, including but not limited to backend systems, 
            frontend interfaces, AI components, and proprietary code, are the property of Michael Miscanuk. 
            This protection extends to the unique implementation, architecture, and integration of these components.
          </p>
        </section>

        <section>
          <h2 className="text-2xl font-semibold text-gray-800 mb-4">2. Third-Party Data</h2>
          <p className="text-gray-700">
            This application incorporates data from the{' '}
            <a 
              href="https://csu.gov.cz/podminky_pro_vyuzivani_a_dalsi_zverejnovani_statistickych_udaju_csu" 
              target="_blank" 
              rel="noopener noreferrer"
              className="text-blue-600 hover:text-blue-800 underline"
            >
              Czech Statistical Office (ČSÚ/CZSU)
            </a>
            . This data remains the intellectual property of ČSÚ and is used in accordance with their terms and conditions. 
            Users must comply with ČSÚ&apos;s licensing requirements for any reuse of their data.
          </p>
        </section>

        <section>
          <h2 className="text-2xl font-semibold text-gray-800 mb-4">3. Permissions and Restrictions</h2>
          <p className="text-gray-700 mb-3">You may not:</p>
          <ul className="list-disc list-inside space-y-2 text-gray-700 ml-4">
            <li>Copy, reproduce, or distribute the source code</li>
            <li>Create derivative works based on this application</li>
            <li>Use the application or its components for commercial purposes without explicit written permission</li>
            <li>Scrape, data mine, or automate access to this service</li>
          </ul>
        </section>

        <section>
          <h2 className="text-2xl font-semibold text-gray-800 mb-4">4. Disclaimer of Warranty</h2>
          <p className="text-gray-700">
            This application is provided &quot;as is&quot; without any warranties. We are not responsible for the 
            accuracy or completeness of third-party data. Refer to official{' '}
            <a 
              href="https://csu.gov.cz/podminky_pro_vyuzivani_a_dalsi_zverejnovani_statistickych_udaju_csu" 
              target="_blank" 
              rel="noopener noreferrer"
              className="text-blue-600 hover:text-blue-800 underline"
            >
              ČSÚ publications
            </a>
            {' '}for authoritative data.
          </p>
        </section>

        <section>
          <h2 className="text-2xl font-semibold text-gray-800 mb-4">5. Third-Party Components</h2>
          <p className="text-gray-700">
            This application uses standard development frameworks and libraries, each governed by their respective 
            licenses. These do not constitute a transfer of rights for our proprietary implementation.
          </p>
        </section>

        <section className="pt-6 border-t border-gray-200">
          <p className="text-gray-700">
            <strong>For permissions or questions:</strong>{' '}
            <a href="mailto:michael.miscanuk@gmail.com" className="text-blue-600 hover:text-blue-800 underline">
              michael.miscanuk@gmail.com
            </a>
          </p>
        </section>
      </div>
    </div>
  );
}
