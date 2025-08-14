import os
import asyncio
import streamlit as st
from langchain.chains import RetrievalQA
from langchain.schema import SystemMessage, HumanMessage, AIMessage
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.prompts import PromptTemplate

# 非同期関数を同期的に実行するユーティリティ
def run_async(func, *args, **kwargs):
    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
    return loop.run_until_complete(func(*args, **kwargs))

# FAISSデータベースの読み込み
def load_db(embeddings):
    return FAISS.load_local('faiss_store', embeddings, allow_dangerous_deserialization=True)

# Streamlitページ初期化
def init_page():
    st.set_page_config(
        page_title='オリジナルチャットボット',
        page_icon='🧑‍💻',
    )

# メイン処理
def main():
    init_page()

    # Google APIキーの取得
    api_key = os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        raise ValueError("GOOGLE_API_KEYが設定されていません。.envファイルを確認してください。")

    # EmbeddingsとLLMの初期化（同期的に扱える場合はそのまま）
    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/embedding-001",
        google_api_key=api_key
    )

    llm = ChatGoogleGenerativeAI(
        model="gemini-1.5-flash",
        temperature=0.0,
        max_retries=2,
        google_api_key=api_key
    )

    db = load_db(embeddings)

    # プロンプトテンプレートの定義
    prompt_template = """
    あなたは、「健康危機管理に携わる地方公共団体等機関職員」専用のチャットボットです。
    背景情報を参考に、質問に対して期間職員になりきって、質問に回答してくだい。

    健康危機管理に携わる地方公共団体等機関に全く関係のない質問と思われる質問に関しては、「健康危機管理に携わる地方公共団体等機関に関係することについて聞いてください」と答えてください。

    以下の背景情報を参照してください。情報がなければ、その内容については言及しないでください。
    # 背景情報
    {context}

    # 質問
    {question}
    """

    PROMPT = PromptTemplate(
        template=prompt_template,
        input_variables=["context", "question"]
    )

    # RetrievalQAチェーンの構築
    qa = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=db.as_retriever(search_kwargs={"k": 2}),
        return_source_documents=True,
        chain_type_kwargs={"prompt": PROMPT}
    )

    # チャット履歴の初期化
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # ユーザー入力の処理
    if user_input := st.chat_input('質問しよう！'):
        # 過去のメッセージ表示
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        with st.chat_message('user'):
            st.markdown(user_input)
        st.session_state.messages.append({"role": "user", "content": user_input})

        with st.chat_message('assistant'):
            with st.spinner('Gemini is typing ...'):
                # 非同期関数を同期的に実行
                response = run_async(qa.invoke, user_input)
            st.markdown(response['result'])

            # 参考元の表示
            doc_urls = []
            for doc in response["source_documents"]:
                url = doc.metadata.get("source_url")
                if url and url not in doc_urls:
                    doc_urls.append(url)
                    st.markdown(f"参考元：{url}")

        st.session_state.messages.append({"role": "assistant", "content": response["result"]})

# 実行
if __name__ == "__main__":
    main()
