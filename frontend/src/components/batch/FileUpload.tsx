import { useState, useCallback, useRef } from 'react';

interface FileUploadProps {
  onFileSelect: (file: File) => void;
  disabled?: boolean;
}

export function FileUpload({ onFileSelect, disabled }: FileUploadProps) {
  const [isDragOver, setIsDragOver] = useState(false);
  const inputRef = useRef<HTMLInputElement>(null);

  const handleFile = useCallback(
    (file: File) => {
      const name = file.name.toLowerCase();
      if (!name.endsWith('.csv') && !name.endsWith('.json')) {
        alert('Nieobsługiwany format. Użyj CSV lub JSON.');
        return;
      }
      onFileSelect(file);
    },
    [onFileSelect],
  );

  const handleDrop = useCallback(
    (e: React.DragEvent) => {
      e.preventDefault();
      setIsDragOver(false);
      const file = e.dataTransfer.files[0];
      if (file) handleFile(file);
    },
    [handleFile],
  );

  return (
    <div
      onDragOver={(e) => {
        e.preventDefault();
        setIsDragOver(true);
      }}
      onDragLeave={() => setIsDragOver(false)}
      onDrop={handleDrop}
      onClick={() => inputRef.current?.click()}
      className={`cursor-pointer rounded-lg border-2 border-dashed p-6 text-center transition ${
        isDragOver ? 'border-blue-400 bg-blue-900/20' : 'border-gray-600 hover:border-gray-400'
      } ${disabled ? 'pointer-events-none opacity-50' : ''}`}
    >
      <p className="text-sm text-gray-400">Przeciągnij plik lub kliknij</p>
      <p className="mt-1 text-xs text-gray-500">CSV, JSON</p>
      <input
        ref={inputRef}
        type="file"
        accept=".csv,.json"
        className="hidden"
        onChange={(e) => {
          const file = e.target.files?.[0];
          if (file) handleFile(file);
          e.target.value = '';
        }}
      />
    </div>
  );
}
