"""
Unit tests for chatbot functions.
Run with: python -m pytest tests/test_chatbot.py -v
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'tomato'))

import pytest
from app import (
    check_faq_response, 
    estimate_tokens,
    CHAT_MAX_QUESTION_LENGTH,
    FAQ_RESPONSES
)


class TestFAQCache:
    """Test FAQ response matching"""
    
    def test_exact_match(self):
        """Test exact FAQ keyword match"""
        response = check_faq_response("cà chua là gì")
        assert response is not None
        assert "Solanum lycopersicum" in response
    
    def test_partial_match(self):
        """Test partial keyword match"""
        response = check_faq_response("triệu chứng bệnh cháy sớm")
        assert response is not None
        assert "cháy sớm" in response.lower()
    
    def test_no_match(self):
        """Test no match returns None"""
        response = check_faq_response("xyz abc completely unrelated")
        assert response is None
    
    def test_case_insensitive(self):
        """Test case insensitive matching"""
        response1 = check_faq_response("CÀ CHUA")
        response2 = check_faq_response("cà chua")
        assert response1 == response2


class TestInputValidation:
    """Test input validation"""
    
    def test_max_question_length(self):
        """Test max question length constant"""
        assert CHAT_MAX_QUESTION_LENGTH == 500
        assert isinstance(CHAT_MAX_QUESTION_LENGTH, int)
    
    def test_token_estimation(self):
        """Test token estimation function"""
        text = "Hello world"  # 11 chars
        tokens = estimate_tokens(text)
        assert tokens == 2  # 11 // 4 = 2
        
        long_text = "a" * 400
        assert estimate_tokens(long_text) == 100


class TestFAQContent:
    """Test FAQ responses quality"""
    
    def test_all_faqs_non_empty(self):
        """Test all FAQ responses are non-empty"""
        for key, answer in FAQ_RESPONSES.items():
            assert len(answer) > 20, f"FAQ '{key}' too short"
            assert isinstance(answer, str)
    
    def test_common_questions_covered(self):
        """Test common questions are in FAQ"""
        common_questions = [
            "cà chua là gì",
            "bệnh cháy sớm",
            "phòng ngừa",
            "triệu chứng"
        ]
        for q in common_questions:
            assert any(q in key for key in FAQ_RESPONSES.keys()), \
                f"Common question '{q}' not in FAQ"


class TestConstants:
    """Test configuration constants"""
    
    def test_chat_constants_exist(self):
        """Test all chat constants are defined"""
        from app import (
            CHAT_MAX_QUESTION_LENGTH,
            CHAT_MIN_ANSWER_LENGTH,
            CHAT_MAX_TOKENS,
            CHAT_API_TIMEOUT,
            CHAT_RATE_LIMIT_PER_MINUTE,
            CHAT_RETRY_ATTEMPTS,
            CHAT_RETRY_DELAY,
            CACHE_TTL
        )
        
        assert CHAT_MAX_QUESTION_LENGTH > 0
        assert CHAT_MIN_ANSWER_LENGTH > 0
        assert CHAT_MAX_TOKENS > 0
        assert CHAT_API_TIMEOUT > 0
        assert CHAT_RATE_LIMIT_PER_MINUTE > 0
        assert CHAT_RETRY_ATTEMPTS > 0
        assert CHAT_RETRY_DELAY > 0
        assert CACHE_TTL > 0


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
