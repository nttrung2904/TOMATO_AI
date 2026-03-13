"""Tests for advanced community social features."""

import json
import sys
from pathlib import Path


sys.path.insert(0, str(Path(__file__).parent.parent / 'tomato'))

import app as tomato_app


def write_jsonl(file_path, rows):
    file_path.parent.mkdir(parents=True, exist_ok=True)
    with open(file_path, 'w', encoding='utf-8') as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + '\n')


class TestSocialTopics:
    def test_extract_post_topics_prefers_relevant_keywords(self):
        topics = tomato_app.extract_post_topics(
            'Mình vừa xử lý bệnh đốm lá rất nặng #domla và chuẩn bị thuhoach vụ mới.'
        )

        assert 'domla' in topics
        assert 'thuhoach' in topics
        assert 'mình' not in topics


class TestSocialFeed:
    def test_build_community_feed_prioritizes_followed_and_interest_posts(self, monkeypatch):
        posts = [
            {
                'id': 'POST001',
                'user_id': 'friend-1',
                'user_name': 'Friend One',
                'content': 'Kinh nghiệm xử lý bệnh đốm lá #domla rất hiệu quả',
                'topics': ['domla'],
                'likes': 1,
                'comments': 1,
                'shares': 0,
                'saves': 0,
                'created_at': '2026-03-12T10:00:00'
            },
            {
                'id': 'POST002',
                'user_id': 'expert-1',
                'user_name': 'Expert',
                'content': 'Checklist thuhoach để trái đồng đều #thuhoach',
                'topics': ['thuhoach'],
                'likes': 6,
                'comments': 3,
                'shares': 1,
                'saves': 2,
                'created_at': '2026-03-12T11:00:00'
            },
            {
                'id': 'POST003',
                'user_id': 'other-1',
                'user_name': 'Other',
                'content': 'Bài viết chung chung về thời tiết',
                'topics': ['thoitiet'],
                'likes': 0,
                'comments': 0,
                'shares': 0,
                'saves': 0,
                'created_at': '2026-03-12T09:00:00'
            }
        ]

        monkeypatch.setattr(
            tomato_app,
            'load_user_follows',
            lambda: [{'follower_id': 'user-1', 'following_id': 'friend-1'}]
        )
        monkeypatch.setattr(
            tomato_app,
            'load_post_likes',
            lambda: [{'user_id': 'user-1', 'post_id': 'POST002'}]
        )
        monkeypatch.setattr(
            tomato_app,
            'load_saved_posts',
            lambda user_id=None: [{'user_id': 'user-1', 'post_id': 'POST002'}] if user_id in (None, 'user-1') else []
        )

        feed = tomato_app.build_community_feed(posts, current_user_id='user-1', feed_type='for_you', limit=3)

        assert feed[0]['id'] == 'POST002'
        assert {post['id'] for post in feed[:2]} == {'POST001', 'POST002'}


class TestSavedPosts:
    def test_toggle_saved_post_updates_post_counter(self, monkeypatch, tmp_path):
        monkeypatch.setattr(tomato_app, 'BASE_DIR', tmp_path)

        write_jsonl(
            tmp_path / 'data' / 'posts.jsonl',
            [{
                'id': 'POST001',
                'user_id': 'user-2',
                'user_name': 'Tester',
                'content': 'Bài viết cần lưu',
                'likes': 0,
                'comments': 0,
                'shares': 0,
                'saves': 0,
                'created_at': '2026-03-10T08:00:00',
                'updated_at': '2026-03-10T08:00:00'
            }]
        )

        is_saved, saves_count = tomato_app.toggle_saved_post('POST001', 'user-1')
        assert is_saved is True
        assert saves_count == 1

        posts = tomato_app.load_posts()
        assert posts[0]['saves'] == 1

        is_saved, saves_count = tomato_app.toggle_saved_post('POST001', 'user-1')
        assert is_saved is False
        assert saves_count == 0

        posts = tomato_app.load_posts()
        assert posts[0]['saves'] == 0