from __future__ import annotations


_TRADITIONAL_TO_SIMPLIFIED = str.maketrans(
    {
        "會": "会",
        "對": "对",
        "機": "机",
        "進": "进",
        "訪": "访",
        "這": "这",
        "個": "个",
        "們": "们",
        "確": "确",
        "認": "认",
        "產": "产",
        "標": "标",
        "權": "权",
        "戶": "户",
        "塊": "块",
        "兒": "儿",
        "討": "讨",
        "導": "导",
        "問": "问",
        "麼": "么",
        "聽": "听",
        "開": "开",
        "關": "关",
        "說": "说",
        "話": "话",
        "語": "语",
        "聲": "声",
        "間": "间",
        "時": "时",
        "長": "长",
        "數": "数",
        "據": "据",
        "錄": "录",
        "實": "实",
        "測": "测",
        "試": "试",
        "輸": "输",
        "處": "处",
        "發": "发",
        "觸": "触",
        "後": "后",
        "續": "续",
        "當": "当",
        "為": "为",
        "應": "应",
        "該": "该",
        "顯": "显",
        "從": "从",
        "與": "与",
        "復": "复",
        "別": "别",
        "雜": "杂",
        "體": "体",
        "簡": "简",
        "轉": "转",
        "換": "换",
        "單": "单",
        "雙": "双",
        "邊": "边",
        "裡": "里",
        "嗎": "吗",
        "沒": "没",
        "現": "现",
        "線": "线",
        "網": "网",
        "類": "类",
        "種": "种",
        "樣": "样",
        "檔": "档",
        "報": "报",
        "錯": "错",
        "讓": "让",
        "備": "备",
        "訓": "训",
        "練": "练",
        "習": "习",
        "庫": "库",
        "國": "国",
        "愛": "爱",
        "爲": "为",
        "妳": "你",
    }
)


def normalize_text(text: str, simplify_chinese: bool) -> str:
    text = text.strip()
    if not simplify_chinese:
        return text
    return to_simplified_chinese(text)


def to_simplified_chinese(text: str) -> str:
    try:
        from opencc import OpenCC

        return OpenCC("t2s").convert(text)
    except Exception:
        pass
    try:
        from zhconv import convert

        return convert(text, "zh-cn")
    except Exception:
        pass
    return text.translate(_TRADITIONAL_TO_SIMPLIFIED)
