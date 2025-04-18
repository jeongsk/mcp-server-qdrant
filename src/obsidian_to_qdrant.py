#!/usr/bin/env python3
"""
옵시디언 노트를 Qdrant 벡터 데이터베이스에 저장하는 스크립트

이 스크립트는 옵시디언 경로를 환경변수로 받아서 모든 마크다운 파일을 읽고,
텍스트를 적절한 크기로 청킹한 후 임베딩을 생성하여 Qdrant에 저장합니다.
"""

import os
import glob
import logging
import asyncio
from typing import List, Dict, Any, Optional
from pathlib import Path
from dotenv import load_dotenv

from mcp_server_qdrant.embeddings.factory import create_embedding_provider
from mcp_server_qdrant.qdrant import QdrantConnector, Entry
from mcp_server_qdrant.settings import (
    EmbeddingProviderSettings,
    QdrantSettings,
)

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

def load_markdown_files(obsidian_path: str) -> List[Dict[str, Any]]:
    """
    옵시디언 경로에서 모든 마크다운 파일을 로드합니다.
    
    Args:
        obsidian_path: 옵시디언 볼트 경로
        
    Returns:
        마크다운 파일 정보 목록 (경로, 내용, 메타데이터)
    """
    markdown_files = []
    
    # 모든 마크다운 파일 경로 찾기
    md_file_paths = glob.glob(os.path.join(obsidian_path, "**/*.md"), recursive=True)
    logger.info(f"총 {len(md_file_paths)}개의 마크다운 파일을 찾았습니다.")
    
    for file_path in md_file_paths:
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
                
            # 파일 경로를 옵시디언 볼트 루트 기준 상대 경로로 변환
            relative_path = os.path.relpath(file_path, obsidian_path)
            
            # 파일명 추출 (확장자 제외)
            filename = os.path.splitext(os.path.basename(file_path))[0]
            
            markdown_files.append({
                "path": relative_path,
                "filename": filename,
                "content": content,
                "metadata": {
                    "path": relative_path,
                    "filename": filename
                }
            })
        except Exception as e:
            logger.error(f"파일 '{file_path}' 처리 중 오류 발생: {e}")
    
    return markdown_files

def chunk_markdown(markdown_files: List[Dict[str, Any]], chunk_size: int = 1000, chunk_overlap: int = 200) -> List[Dict[str, Any]]:
    """
    마크다운 파일을 적절한 크기로 청킹합니다.
    
    Args:
        markdown_files: 마크다운 파일 정보 목록
        chunk_size: 각 청크의 최대 문자 수
        chunk_overlap: 청크 간 겹치는 문자 수
        
    Returns:
        청킹된 텍스트 목록
    """
    chunks = []
    
    for md_file in markdown_files:
        content = md_file["content"]
        
        # 파일이 청크 크기보다 작으면 그대로 사용
        if len(content) <= chunk_size:
            chunks.append({
                "content": content,
                "metadata": {
                    **md_file["metadata"],
                    "chunk_index": 0,
                    "total_chunks": 1
                }
            })
            continue
        
        # 청크 나누기
        current_index = 0
        chunk_index = 0
        
        while current_index < len(content):
            # 청크 끝 위치 계산
            chunk_end = min(current_index + chunk_size, len(content))
            
            # 청크가 문장 중간에서 끝나지 않도록 조정
            if chunk_end < len(content):
                # 문단 끝, 문장 끝, 또는 공백에서 끊기
                for separator in ["\n\n", ".\n", ". ", "\n", ". ", " "]:
                    pos = content.rfind(separator, current_index, chunk_end)
                    if pos != -1:
                        chunk_end = pos + len(separator)
                        break
            
            chunk_text = content[current_index:chunk_end]
            
            # 청크가 너무 작으면 건너뛰기
            if len(chunk_text.strip()) > 50:  # 최소 50자 이상인 청크만 저장
                chunks.append({
                    "content": chunk_text,
                    "metadata": {
                        **md_file["metadata"],
                        "chunk_index": chunk_index,
                        "chunk_start": current_index,
                        "chunk_end": chunk_end
                    }
                })
                chunk_index += 1
            
            # 다음 청크의 시작 위치 (오버랩 고려)
            current_index = chunk_end - chunk_overlap
            if current_index <= current_index:  # 진전이 없으면
                current_index = chunk_end  # 오버랩 없이 진행
    
    logger.info(f"총 {len(chunks)}개의 청크를 생성했습니다.")
    return chunks

async def process_obsidian_vault():
    """
    옵시디언 볼트를 처리하고 Qdrant에 저장하는 메인 함수
    """
    # .env 파일 로드
    load_dotenv()
    
    # 환경 변수 로드
    obsidian_path = os.getenv("OBSIDIAN_PATH")
    if not obsidian_path:
        logger.error("OBSIDIAN_PATH 환경 변수가 설정되지 않았습니다.")
        return
    
    collection_name = os.getenv("COLLECTION_NAME", "obsidian-notes")
    embedding_model = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
    chunk_size = int(os.getenv("CHUNK_SIZE", "1000"))
    chunk_overlap = int(os.getenv("CHUNK_OVERLAP", "200"))
    
    # Qdrant 설정
    qdrant_url = os.getenv("QDRANT_URL")
    qdrant_api_key = os.getenv("QDRANT_API_KEY")
    qdrant_local_path = os.getenv("QDRANT_LOCAL_PATH")
    
    if not (qdrant_url or qdrant_local_path):
        logger.error("QDRANT_URL 또는 QDRANT_LOCAL_PATH 환경 변수가 설정되어야 합니다.")
        return
    
    # 임베딩 제공자 생성
    embedding_settings = EmbeddingProviderSettings(model_name=embedding_model)
    embedding_provider = create_embedding_provider(embedding_settings)
    
    # Qdrant 커넥터 생성
    qdrant_settings = QdrantSettings(
        location=qdrant_url,
        api_key=qdrant_api_key,
        collection_name=collection_name,
        local_path=qdrant_local_path
    )
    
    qdrant_connector = QdrantConnector(
        qdrant_url,
        qdrant_api_key,
        collection_name,
        embedding_provider,
        qdrant_local_path
    )
    
    # 옵시디언 볼트에서 마크다운 파일 로드
    logger.info(f"옵시디언 볼트 경로: {obsidian_path}")
    markdown_files = load_markdown_files(obsidian_path)
    logger.info(f"{len(markdown_files)}개의 마크다운 파일을 로드했습니다.")
    
    # 마크다운 파일 청킹
    chunks = chunk_markdown(markdown_files, chunk_size, chunk_overlap)
    
    # Qdrant에 저장
    logger.info(f"청크를 Qdrant 컬렉션 '{collection_name}'에 저장합니다...")
    
    for i, chunk in enumerate(chunks):
        try:
            entry = Entry(content=chunk["content"], metadata=chunk["metadata"])
            await qdrant_connector.store(entry)
            
            if (i + 1) % 100 == 0 or (i + 1) == len(chunks):
                logger.info(f"진행 상황: {i + 1}/{len(chunks)} 청크 처리 완료")
        except Exception as e:
            logger.error(f"청크 저장 중 오류 발생: {e}")
    
    logger.info("모든 옵시디언 노트가 Qdrant에 성공적으로 저장되었습니다.")

if __name__ == "__main__":
    asyncio.run(process_obsidian_vault())
