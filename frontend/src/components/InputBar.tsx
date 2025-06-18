import React, { ChangeEvent, FormEvent, forwardRef } from "react"

interface InputBarProps {
    currentMessage: string;
    setCurrentMessage: (msg: string) => void;
    onSubmit: (e: FormEvent<HTMLFormElement>) => void;
    isLoading?: boolean;
}

const InputBar = forwardRef<HTMLInputElement, InputBarProps>(({ currentMessage, setCurrentMessage, onSubmit, isLoading = false }, ref) => {

    const handleChange = (e: ChangeEvent<HTMLInputElement>) => {
        setCurrentMessage(e.target.value)
    }

    return (
        <form onSubmit={onSubmit} className="p-4 bg-white">
            <div className="flex items-center bg-[#F9F9F5] rounded-full p-3 shadow-md border border-gray-200 relative focus-within:ring-2 focus-within:ring-blue-500 focus-within:border-blue-500 focus-within:z-10 transition-all duration-200">
                <input
                    ref={ref}
                    type="text"
                    placeholder="Type a message"
                    aria-label="Type a message"
                    value={currentMessage}
                    onChange={handleChange}
                    className="flex-grow px-4 py-2 bg-transparent focus:outline-none text-gray-700"
                    disabled={isLoading}
                />
                <button
                    type="submit"
                    aria-label="Send message"
                    className="bg-gradient-to-r from-blue-500 to-blue-400 hover:from-blue-600 hover:to-blue-500 rounded-full p-3 ml-2 shadow-md transition-all duration-200 group relative z-0"
                    disabled={isLoading}
                >
                    <svg className="w-6 h-6 text-white transform rotate-45 group-hover:scale-110 transition-transform duration-200" fill="none" stroke="currentColor" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M12 19l9 2-9-18-9 18 9-2zm0 0v-8"></path>
                    </svg>
                </button>
            </div>
        </form>
    )
})

InputBar.displayName = 'InputBar'

export default InputBar