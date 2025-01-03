import os
from langchain.text_splitter import RecursiveCharacterTextSplitter

# 파일 경로 설정
file_path_jju = r'C:\Users\nnnee\OneDrive - 전주대학교\2024 2학기\학교\비트교육\html_jju.txt'

# 파일 경로 확인
if not os.path.exists(file_path_jju):
    print(f"파일이 존재하지 않습니다: {file_path_jju}")
else:
    # 파일 내용을 읽어오기
    with open(file_path_jju, 'r', encoding='utf-8') as f:
        file_content = f.read()

    # HTML 텍스트 분할기 생성
    html_splitter = RecursiveCharacterTextSplitter(
        chunk_size=60,  # 청크 크기를 60으로 설정
        chunk_overlap=0  # 청크 간 중복되는 부분이 없도록 설정
    )

    # HTML 텍스트를 분할하여 문서 생성
    html_docs = html_splitter.create_documents([file_content])

    # 분할된 문서 출력
    for idx, doc in enumerate(html_docs):
        print(f"문서 {idx + 1}:\n{doc.page_content}\n")