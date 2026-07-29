#!/usr/bin/env python3
"""
Testes sandbox para melhorias P0 de completude (retrieval + prompts).

Executar:
    cd RAG-ANTT
    source venv/bin/activate
    python test_completude_p0.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from langchain_core.documents import Document

from antt_rag_unified import (
    _INSTRUCOES_COMPLETUDE,
    _aplicar_limite_k_preservando_prioritarios,
    _detectar_referencia_documento,
    _dividir_por_estrutura,
    _preparar_contexto_resposta,
    _resolver_caminho_documento,
    carregar_vectorstore_com_provider,
    pesquisar_documentos,
)


def test_instrucoes_completude_presentes():
    assert "NAO generalize" in _INSTRUCOES_COMPLETUDE or "Nao generalize" in _INSTRUCOES_COMPLETUDE
    assert "fase" in _INSTRUCOES_COMPLETUDE.lower()
    assert "RESPOSTA DIRETA" in _INSTRUCOES_COMPLETUDE
    assert "EQUIVALENCIA TERMINOLOGICA" in _INSTRUCOES_COMPLETUDE
    assert "LACUNAS PONTUAIS" in _INSTRUCOES_COMPLETUDE
    print("OK: instrucoes de completude definidas")


def test_detectar_referencia_inm_34():
    query = (
        "Quais sao os valores limites de IRI definidos pela "
        "Instrucao Normativa 34/2024?"
    )
    refs = _detectar_referencia_documento(query)
    assert len(refs) >= 1, f"Nenhuma referencia detectada: {refs}"
    assert refs[0]["tipo"] == "INM"
    assert refs[0]["numero"] == "34"
    assert refs[0]["ano"] == "2024"
    print(f"OK: referencia detectada {refs[0]}")


def test_limite_k_preserva_prioritarios():
    docs = []
    for i in range(5):
        docs.append(Document(page_content=f"prior {i}", metadata={"prioritario": True, "chunk": i}))
    for i in range(10):
        docs.append(Document(page_content=f"outro {i}", metadata={"prioritario": False, "chunk": i}))

    resultado = _aplicar_limite_k_preservando_prioritarios(docs, k=6)
    prioritarios = [d for d in resultado if d.metadata.get("prioritario")]
    assert len(prioritarios) == 5, f"Esperado 5 prioritarios, obteve {len(prioritarios)}"
    assert len(resultado) == 6, f"Esperado 6 total, obteve {len(resultado)}"
    print("OK: limite k preserva todos os chunks prioritarios")


def test_tabela_nao_fragmentada():
    texto_tabela = (
        "Art. 1 Texto introdutorio.\n\n"
        "| Pista | Fase | Valor |\n"
        "| --- | --- | --- |\n"
        "| Principal | Recuperacao | 2,7 m/km |\n"
        "| Marginal | Manutencao | 3,0 m/km |\n"
    )
    chunks = _dividir_por_estrutura(texto_tabela, chunk_max=1500)
    tabela_inteira = any("| --- |" in c and "2,7 m/km" in c and "3,0 m/km" in c for c in chunks)
    assert tabela_inteira, f"Tabela fragmentada em {len(chunks)} chunks: {chunks}"
    print(f"OK: tabela preservada em chunk unico ({len(chunks)} chunk(s))")


def test_contexto_inclui_instrucoes_completude():
    docs = [
        Document(
            page_content="| A | B |\n| --- | --- |\n| 1 | 2 |",
            metadata={"nome_tipo": "INM", "numero": "34", "ano": "2024", "chunk": 1, "total_chunks": 1},
        )
    ]
    prompt, template_tipo, corpo = _preparar_contexto_resposta(
        "Quais os limites de IRI?", docs, modelo_usado="deepseek"
    )
    assert _INSTRUCOES_COMPLETUDE.strip() in prompt
    assert template_tipo == "parametros"
    assert corpo is not None
    assert "| A | B |" in corpo
    print(f"OK: contexto inclui instrucoes completude (template={template_tipo})")


def test_pesquisa_inm_34_tem_tabela():
    if not os.path.exists("vectorstore_local"):
        print("SKIP: vectorstore_local nao encontrado (execute reindexacao antes)")
        return

    vs = carregar_vectorstore_com_provider("local")
    query = (
        "Quais sao os valores limites de IRI definidos pela "
        "Instrucao Normativa 34/2024?"
    )
    docs = pesquisar_documentos(query, vs, k=10, embedding_provider="local")

    assert len(docs) > 0, "Nenhum documento retornado"
    prioritarios = [d for d in docs if d.metadata.get("prioritario")]
    com_tabela = [d for d in docs if "| --- |" in d.page_content]
    com_iri = [d for d in docs if "iri" in d.page_content.lower() or "2,7" in d.page_content]

    print(f"  Total chunks: {len(docs)}")
    print(f"  Prioritarios: {len(prioritarios)}")
    print(f"  Com tabela markdown: {len(com_tabela)}")
    print(f"  Com IRI/2,7: {len(com_iri)}")

    assert len(prioritarios) >= 1, "Nenhum chunk prioritario da INM 34"
    assert len(com_tabela) >= 1 or len(com_iri) >= 1, (
        "Contexto sem tabela IRI - verifique OCR/indexacao"
    )
    print("OK: pesquisa INM 34 retorna chunks prioritarios com dados IRI")


def main():
    print("=" * 60)
    print("TESTES P0 COMPLETUDE (sandbox)")
    print("=" * 60)

    test_instrucoes_completude_presentes()
    test_detectar_referencia_inm_34()
    test_limite_k_preserva_prioritarios()
    test_tabela_nao_fragmentada()
    test_contexto_inclui_instrucoes_completude()
    test_pesquisa_inm_34_tem_tabela()

    print("=" * 60)
    print("TODOS OS TESTES PASSARAM")
    print("=" * 60)


if __name__ == "__main__":
    main()
