from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
import os
from langchain_community.document_loaders.pdf import PyPDFLoader
from langchain_community.document_loaders import PyPDFium2Loader
import fitz  
from pdfminer.high_level import extract_text
import pdfplumber

os.environ['OPENAI_API_KEY'] = ''

# PDF 파일 경로
FILE_PATH = r'C:\Users\nnnee\Documents\GitHub\test_jju\02_load\중소벤처기업부_공고_제2024-554호(벤처투자회사_등록말소).pdf'

# 1. PyPDFLoader (PyPDF2 스타일 로딩)
def load_pypdf():
    loader = PyPDFLoader(FILE_PATH)
    docs = loader.load()
    return "\n".join([doc.page_content for doc in docs])

# 2. PyPDFium2Loader
def load_pypdfium2():
    loader = PyPDFium2Loader(FILE_PATH)
    docs = loader.load()
    return "\n".join([doc.page_content for doc in docs])

# 3. PyMuPDF (fitz)로 텍스트 추출
def load_pymupdf():
    doc = fitz.open(FILE_PATH)
    text = ""
    for page_num in range(len(doc)):
        page = doc.load_page(page_num)
        text += page.get_text()
    return text

# 4. pdfminer로 텍스트 추출
def load_pdfminer():
    return extract_text(FILE_PATH)

# 5. pdfplumber로 텍스트 추출
def load_pdfplumber():
    with pdfplumber.open(FILE_PATH) as pdf:
        text = ""
        for page in pdf.pages:
            text += page.extract_text()
    return text

# 각 로더 함수 호출 후 텍스트 추출
docs_pypdf = load_pypdf()
docs_pypdfium2 = load_pypdfium2()
text_pymupdf = load_pymupdf()
text_pdfminer = load_pdfminer()
text_pdfplumber = load_pdfplumber()

# ChatOpenAI 모델 초기화
model = ChatOpenAI(
    model="gpt-4o",
    max_tokens=2048,
    temperature=0.1
)

prompt_template = """
당신은 유용한 어시스턴트입니다. 아래에서 추출된 텍스트를 분석해 주세요:

Text extracted using PyPDFLoader:
{docs_pypdf}

Text extracted using PyPDFium2Loader:
{docs_pypdfium2}

Text extracted using PyMuPDF:
{text_pymupdf}

Text extracted using pdfminer:
{text_pdfminer}

Text extracted using pdfplumber:
{text_pdfplumber}

위 텍스트에서 요약 또는 중요한 정보를 제공해 주세요.

추가로, 위의 추출된 정보를 기반으로 RAG (Retrieval-Augmented Generation) 성능을 평가해주세요. 아래 항목들에 대해 정량적으로 점수를 매겨 주세요:

1. 정확도 (Accuracy) - 추출된 정보의 정확성
2. 정보 추출 정확성 - 중요한 정보가 누락 없이 정확히 추출되었는지
3. 응답의 유창성 - 생성된 응답이 자연스럽고 읽기 쉬운지
4. 문맥 일관성 - 주어진 문맥에 맞춰 일관된 응답을 제공하는지
5. 응답 시간 - 정보 검색과 생성 속도가 적절한지

각 항목에 대해 0~5점 사이로 평가하고, 전체 평균 점수를 계산하고 표로 정리해 주세요.

**출력 형식:**
평가 항목 | 점수 (0~5) | 설명
-----------|-----------|------------------------------------------------------
정확도      | [점수]     | [설명]
정보 추출 정확성 | [점수]     | [설명]
응답의 유창성 | [점수]     | [설명]
문맥 일관성 | [점수]     | [설명]
응답 시간   | [점수]     | [설명]
평균 점수   | [평균]     | 
"""


# 모델에 전달할 프롬프트 생성
prompt = PromptTemplate(
    input_variables=["docs_pypdf", "docs_pypdfium2", "text_pymupdf", "text_pdfminer", "text_pdfplumber"],
    template=prompt_template
)

# 모델과 프롬프트 템플릿을 연결한 체인 생성
chain = prompt | model  # Deprecated된 LLMChain 대신 사용할 새로운 방식

# 체인 실행하여 결과 받기
response = chain.invoke({
    "docs_pypdf": docs_pypdf[:500],  # PyPDFLoader에서 추출한 텍스트 500자
    "docs_pypdfium2": docs_pypdfium2[:500],  # PyPDFium2Loader에서 추출한 텍스트 500자
    "text_pymupdf": text_pymupdf[:500],  # PyMuPDF에서 추출한 텍스트 500자
    "text_pdfminer": text_pdfminer[:500],  # pdfminer에서 추출한 텍스트 500자
    "text_pdfplumber": text_pdfplumber[:500]  # pdfplumber에서 추출한 텍스트 500자
})

# 모델의 응답 출력
print("GPT-4o 모델의 응답:\n", response)