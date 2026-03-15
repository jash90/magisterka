import ReactMarkdown from 'react-markdown';

interface MarkdownMessageProps {
  content: string;
}

/**
 * Renders chat message content as Markdown.
 * Handles **bold**, *italic*, lists, headers, links, inline code, etc.
 */
export function MarkdownMessage({ content }: MarkdownMessageProps) {
  return (
    <ReactMarkdown
      components={{
        p: ({ children }) => <p className="mb-1 last:mb-0">{children}</p>,
        strong: ({ children }) => <strong className="font-bold text-white">{children}</strong>,
        em: ({ children }) => <em className="italic text-gray-300">{children}</em>,
        ul: ({ children }) => <ul className="mb-1 ml-4 list-disc space-y-0.5">{children}</ul>,
        ol: ({ children }) => <ol className="mb-1 ml-4 list-decimal space-y-0.5">{children}</ol>,
        li: ({ children }) => <li className="text-gray-200">{children}</li>,
        h1: ({ children }) => <h1 className="mb-1 text-lg font-bold text-blue-300">{children}</h1>,
        h2: ({ children }) => <h2 className="mb-1 text-base font-bold text-blue-300">{children}</h2>,
        h3: ({ children }) => <h3 className="mb-1 text-sm font-bold text-blue-300">{children}</h3>,
        code: ({ children }) => (
          <code className="rounded bg-gray-600/50 px-1 py-0.5 text-xs text-blue-200">{children}</code>
        ),
        pre: ({ children }) => (
          <pre className="my-1 overflow-x-auto rounded bg-gray-900 p-2 text-xs text-gray-300">{children}</pre>
        ),
        blockquote: ({ children }) => (
          <blockquote className="my-1 border-l-2 border-blue-500 pl-3 text-gray-300">{children}</blockquote>
        ),
        hr: () => <hr className="my-2 border-gray-600" />,
        a: ({ href, children }) => (
          <a href={href} className="text-blue-400 underline hover:text-blue-300" target="_blank" rel="noopener noreferrer">
            {children}
          </a>
        ),
      }}
    >
      {content}
    </ReactMarkdown>
  );
}
