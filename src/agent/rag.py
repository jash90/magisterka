"""
Moduł RAG (Retrieval-Augmented Generation) dla agenta konwersacyjnego.

Implementuje pipeline RAG z użyciem LangChain i ChromaDB
do kontekstualizacji odpowiedzi LLM.
"""

import os
from pathlib import Path
from typing import List, Dict, Any, Optional, TYPE_CHECKING
import json
import logging

# Type checking imports for IDE support
if TYPE_CHECKING:
    from langchain.schema import Document

try:
    from langchain_openai import ChatOpenAI, OpenAIEmbeddings
    from langchain_community.vectorstores import Chroma
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    from langchain.schema import Document, HumanMessage, SystemMessage
    from langchain.prompts import PromptTemplate
    LANGCHAIN_AVAILABLE = True
except ImportError:
    LANGCHAIN_AVAILABLE = False
    Document = Any  # Fallback type for when langchain not available
    logging.warning("LangChain not available. RAG features will be limited.")

from .prompts import (
    SYSTEM_PROMPT_PATIENT_BASIC,
    SYSTEM_PROMPT_PATIENT_ADVANCED,
    SYSTEM_PROMPT_CLINICIAN,
    DISCLAIMER_PATIENT,
    FEATURE_TRANSLATIONS,
    get_risk_level,
    RISK_LEVEL_DESCRIPTIONS
)
from .guardrails import GuardrailsChecker

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class RAGPipeline:
    """
    Pipeline RAG do generowania wyjaśnień z kontekstem.

    Wykorzystuje LangChain i ChromaDB do wzbogacania odpowiedzi LLM
    o kontekst z bazy wiedzy medycznej i wyjaśnień XAI.

    Attributes:
        llm: Model językowy
        embeddings: Model embeddingów
        vectorstore: Baza wektorowa ChromaDB
        guardrails: Checker guardrails
    """

    def __init__(
        self,
        llm_model: str = "gpt-4",
        embedding_model: str = "text-embedding-ada-002",
        temperature: float = 0.3,
        persist_directory: str = "./chroma_db"
    ):
        """
        Inicjalizacja RAG Pipeline.

        Args:
            llm_model: Nazwa modelu LLM
            embedding_model: Nazwa modelu embeddingów
            temperature: Temperatura generacji
            persist_directory: Katalog na bazę wektorową
        """
        self.persist_directory = persist_directory
        self.guardrails = GuardrailsChecker()

        if LANGCHAIN_AVAILABLE:
            api_key = os.getenv("OPENAI_API_KEY")

            if api_key:
                self.llm = ChatOpenAI(
                    model=llm_model,
                    temperature=temperature,
                    api_key=api_key
                )
                self.embeddings = OpenAIEmbeddings(
                    model=embedding_model,
                    api_key=api_key
                )
                self.vectorstore = None
                logger.info(f"RAGPipeline zainicjalizowany z modelem {llm_model}")
            else:
                self.llm = None
                self.embeddings = None
                self.vectorstore = None
                logger.warning("OPENAI_API_KEY nie ustawiony. RAG pipeline w trybie offline.")
        else:
            self.llm = None
            self.embeddings = None
            self.vectorstore = None
            logger.warning("LangChain niedostępny. Używam trybu fallback.")

    def create_knowledge_base(
        self,
        documents: List[Dict[str, str]],
        chunk_size: int = 1000,
        chunk_overlap: int = 200
    ) -> None:
        """
        Utwórz bazę wiedzy z dokumentów.

        Args:
            documents: Lista dokumentów {'content': str, 'metadata': dict}
            chunk_size: Rozmiar chunków
            chunk_overlap: Nakładanie chunków
        """
        if not LANGCHAIN_AVAILABLE or self.embeddings is None:
            logger.warning("Tworzenie bazy wiedzy niedostępne bez LangChain/API key")
            return

        # Konwertuj do Document objects
        docs = []
        for doc in documents:
            docs.append(Document(
                page_content=doc['content'],
                metadata=doc.get('metadata', {})
            ))

        # Split documents
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )
        splits = text_splitter.split_documents(docs)

        # Utwórz vectorstore
        self.vectorstore = Chroma.from_documents(
            documents=splits,
            embedding=self.embeddings,
            persist_directory=self.persist_directory
        )

        logger.info(f"Utworzono bazę wiedzy z {len(splits)} chunków")

    def add_medical_knowledge(self) -> None:
        """Dodaj podstawową wiedzę medyczną o zapaleniu naczyń."""
        medical_docs = [
            {
                'content': """
                Zapalenie naczyń (vasculitis) to grupa chorób charakteryzujących się
                zapaleniem ścian naczyń krwionośnych. Może prowadzić do uszkodzenia
                narządów zaopatrywanych przez dotknięte naczynia.

                Główne typy zapalenia naczyń:
                - Ziarniniakowatość z zapaleniem naczyń (GPA, dawniej Wegenera)
                - Mikroskopowe zapalenie naczyń (MPA)
                - Eozynofilowa ziarniniakowatość z zapaleniem naczyń (EGPA, Churg-Strauss)
                - Zapalenie naczyń związane z ANCA (AAV)

                Czynniki prognostyczne:
                - Wiek w momencie rozpoznania
                - Zajęcie nerek
                - Zajęcie serca
                - Zajęcie OUN
                - Liczba zajętych narządów
                """,
                'metadata': {'topic': 'vasculitis_overview', 'type': 'medical_knowledge'}
            },
            {
                'content': """
                Wskaźniki laboratoryjne w zapaleniu naczyń:

                CRP (białko C-reaktywne):
                - Marker stanu zapalnego
                - Norma: <5 mg/L
                - Podwyższone wartości wskazują na aktywność choroby

                Kreatynina:
                - Wskaźnik funkcji nerek
                - Norma: 60-110 μmol/L
                - Podwyższone wartości mogą wskazywać na zajęcie nerek

                ANCA (przeciwciała przeciwko cytoplazmie neutrofilów):
                - PR3-ANCA: często w GPA
                - MPO-ANCA: często w MPA
                """,
                'metadata': {'topic': 'laboratory_markers', 'type': 'medical_knowledge'}
            },
            {
                'content': """
                Leczenie zapalenia naczyń:

                Indukcja remisji:
                - Glikokortykosteroidy (prednizon, metyloprednizolon)
                - Cyklofosfamid lub rytuksymab

                Podtrzymanie remisji:
                - Azatiopryna, metotreksat lub rytuksymab
                - Stopniowa redukcja sterydów

                Terapie w ciężkich przypadkach:
                - Plazmafereza - w krwotoku płucnym lub ciężkiej niewydolności nerek
                - Dializa - przy schyłkowej niewydolności nerek

                Powikłania leczenia:
                - Infekcje oportunistyczne
                - Osteoporoza
                - Cukrzyca posterydowa
                - Powikłania sercowo-naczyniowe
                """,
                'metadata': {'topic': 'treatment', 'type': 'medical_knowledge'}
            }
        ]

        self.create_knowledge_base(medical_docs)

    def retrieve_context(
        self,
        query: str,
        k: int = 3
    ) -> List[Document]:
        """
        Pobierz relevantny kontekst z bazy wiedzy.

        Args:
            query: Zapytanie
            k: Liczba dokumentów do pobrania

        Returns:
            Lista relevantnych dokumentów
        """
        if self.vectorstore is None:
            return []

        try:
            docs = self.vectorstore.similarity_search(query, k=k)
            return docs
        except Exception as e:
            logger.error(f"Błąd podczas retrieval: {e}")
            return []

    def generate_response(
        self,
        query: str,
        xai_explanation: Dict[str, Any],
        health_literacy: str = 'basic',
        use_rag: bool = True
    ) -> str:
        """
        Wygeneruj odpowiedź z użyciem RAG.

        Args:
            query: Pytanie użytkownika
            xai_explanation: Wyjaśnienie z modułu XAI
            health_literacy: Poziom ('basic', 'advanced', 'clinician')
            use_rag: Czy używać RAG

        Returns:
            Wygenerowana odpowiedź
        """
        # Sprawdź guardrails
        guardrail_response = self.guardrails.check_query(query)
        if guardrail_response:
            return guardrail_response

        # Wybierz prompt systemowy
        if health_literacy == 'clinician':
            system_prompt = SYSTEM_PROMPT_CLINICIAN
        elif health_literacy == 'advanced':
            system_prompt = SYSTEM_PROMPT_PATIENT_ADVANCED
        else:
            system_prompt = SYSTEM_PROMPT_PATIENT_BASIC

        # Przygotuj kontekst XAI
        xai_context = self._format_xai_context(xai_explanation, health_literacy)

        # Pobierz kontekst z RAG
        rag_context = ""
        if use_rag and self.vectorstore:
            relevant_docs = self.retrieve_context(query)
            if relevant_docs:
                rag_context = "\n\nKontekst medyczny:\n" + "\n".join([
                    doc.page_content for doc in relevant_docs
                ])

        # Wygeneruj odpowiedź
        if self.llm:
            response = self._generate_with_llm(
                system_prompt=system_prompt,
                query=query,
                xai_context=xai_context,
                rag_context=rag_context
            )
        else:
            response = self._generate_fallback(
                xai_explanation=xai_explanation,
                health_literacy=health_literacy
            )

        # Sprawdź odpowiedź guardrails
        response = self.guardrails.validate_response(response)

        # Dodaj disclaimer
        if health_literacy != 'clinician':
            response += f"\n\n{DISCLAIMER_PATIENT}"

        return response

    def _format_xai_context(
        self,
        explanation: Dict[str, Any],
        health_literacy: str
    ) -> str:
        """Sformatuj kontekst XAI dla promptu."""
        probability = explanation.get('probability_positive', explanation.get('probability', {}).get('Zgon', 0.5))
        risk_level = get_risk_level(probability)

        if health_literacy == 'clinician':
            # Format techniczny
            risk_factors = explanation.get('risk_factors', [])
            protective_factors = explanation.get('protective_factors', [])

            context = f"""
Wyniki analizy XAI:
- Prawdopodobieństwo zgonu: {probability:.1%}
- Poziom ryzyka: {risk_level}

Główne czynniki ryzyka:
{self._format_factors(risk_factors, technical=True)}

Czynniki ochronne:
{self._format_factors(protective_factors, technical=True)}
"""
        else:
            # Format przyjazny dla pacjenta
            risk_factors = explanation.get('risk_factors', [])[:3]
            protective_factors = explanation.get('protective_factors', [])[:3]

            context = f"""
Wyniki analizy:
- Ogólna ocena ryzyka: {RISK_LEVEL_DESCRIPTIONS[risk_level]['patient']}

Czynniki wymagające uwagi:
{self._format_factors(risk_factors, technical=False)}

Pozytywne aspekty:
{self._format_factors(protective_factors, technical=False)}
"""

        return context

    def _format_factors(
        self,
        factors: List,
        technical: bool = False
    ) -> str:
        """Sformatuj listę czynników."""
        if not factors:
            return "- Brak znaczących czynników"

        lines = []
        for factor in factors[:5]:
            if isinstance(factor, dict):
                feature = factor.get('feature', factor.get('variable', ''))
                value = factor.get('shap_value', factor.get('contribution', factor.get('score', 0)))

                if technical:
                    lines.append(f"- {feature}: {value:+.3f}")
                else:
                    translated = FEATURE_TRANSLATIONS.get(feature.split()[0], feature)
                    lines.append(f"- {translated}")
            elif isinstance(factor, tuple):
                feature, value = factor[0], factor[1]
                if technical:
                    lines.append(f"- {feature}: {value:+.3f}")
                else:
                    # Wyodrębnij nazwę cechy
                    for orig_name in FEATURE_TRANSLATIONS:
                        if orig_name in feature:
                            lines.append(f"- {FEATURE_TRANSLATIONS[orig_name]}")
                            break
                    else:
                        lines.append(f"- {feature}")

        return "\n".join(lines)

    def _generate_with_llm(
        self,
        system_prompt: str,
        query: str,
        xai_context: str,
        rag_context: str
    ) -> str:
        """Wygeneruj odpowiedź z LLM."""
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=f"""
Na podstawie poniższych informacji odpowiedz na pytanie pacjenta/lekarza.

{xai_context}

{rag_context}

Pytanie: {query}

Odpowiedz zgodnie z wytycznymi w promptcie systemowym.
""")
        ]

        try:
            response = self.llm.invoke(messages)
            return response.content
        except Exception as e:
            logger.error(f"Błąd LLM: {e}")
            return self._generate_fallback({}, 'basic')

    def _generate_fallback(
        self,
        xai_explanation: Dict[str, Any],
        health_literacy: str
    ) -> str:
        """Wygeneruj odpowiedź fallback bez LLM."""
        probability = xai_explanation.get('probability_positive', 0.5)
        risk_level = get_risk_level(probability)

        if health_literacy == 'clinician':
            return f"""
## Wynik analizy

Prawdopodobieństwo zgonu: {probability:.1%}
Poziom ryzyka: {risk_level.upper()}

{RISK_LEVEL_DESCRIPTIONS[risk_level]['clinician']}

Szczegółowe wyjaśnienie wymaga dostępu do API LLM.
Sprawdź konfigurację OPENAI_API_KEY.
"""
        else:
            return f"""
{RISK_LEVEL_DESCRIPTIONS[risk_level]['patient']}

Zachęcamy do omówienia tych wyników z lekarzem prowadzącym,
który pomoże Ci zrozumieć ich znaczenie w kontekście
Twojego indywidualnego stanu zdrowia.

Uwaga: Szczegółowe wyjaśnienie wymaga połączenia z serwerem.
Prosimy o kontakt z pomocą techniczną.
"""

    def chat(
        self,
        messages_history: List[Dict[str, str]],
        xai_explanation: Dict[str, Any],
        health_literacy: str = 'basic'
    ) -> str:
        """
        Kontynuuj rozmowę z historią.

        Args:
            messages_history: Historia rozmowy [{'role': 'user/assistant', 'content': str}]
            xai_explanation: Wyjaśnienie XAI
            health_literacy: Poziom health literacy

        Returns:
            Odpowiedź asystenta
        """
        if not messages_history:
            return "Jak mogę Ci pomóc zrozumieć wyniki analizy?"

        # Ostatnie pytanie użytkownika
        last_query = messages_history[-1].get('content', '')

        return self.generate_response(
            query=last_query,
            xai_explanation=xai_explanation,
            health_literacy=health_literacy
        )

    def save_conversation(
        self,
        conversation: List[Dict[str, str]],
        filepath: str
    ) -> None:
        """Zapisz rozmowę do pliku JSON."""
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(conversation, f, ensure_ascii=False, indent=2)
        logger.info(f"Rozmowa zapisana do {filepath}")

    def load_conversation(
        self,
        filepath: str
    ) -> List[Dict[str, str]]:
        """Wczytaj rozmowę z pliku JSON."""
        with open(filepath, 'r', encoding='utf-8') as f:
            return json.load(f)
