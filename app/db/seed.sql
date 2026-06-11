BEGIN;

-- ── 역할 (role) ─────────────────────────────────────────────
INSERT INTO public.role (role_name, description) VALUES
    ('실무 담당자', '문서 업로드/요약/RAG 질의/기안 작성'),
    ('결재권자',   '상신된 기안 승인/반려'),
    ('관리자',     '사용자/권한 관리, 통계 조회')
ON CONFLICT (role_name) DO NOTHING;

-- ── 사용자 (users) ─────────────────────────────────────────────
INSERT INTO public.users -- 초기 비밀번호 000000
    (user_email, user_password, user_name, created_at)
VALUES
    -- 1) 시스템 관리자
    ('admin@agency.go.kr',
     '$5$rounds=535000$eWXQtRpuhm6Pp4Ta$iU/8OnPLQ7T6Jr0ExwXMP7uNvdllnabDN/u3e7WU8d8',
      '김관리', '2026-06-01 09:00:00.000001+09'),
 
    -- 2) 박과장 — 결재권자 겸 실무 (프론트엔드 박과장 케이스)
    ('park.jihun@agency.go.kr',
     '$5$rounds=535000$b7Ulg2rRKCGqdJgG$i9ubG0MFZANEMAq2s/zLJBW7g5X3T.JhkIHZT21Mm77',
      '박지훈', '2026-06-01 09:00:00.000002+09');
 
 -- ── 사용자역할 (user_role) ─────────────────────────────────────────────
INSERT INTO public.user_role 
(user_id, role_id)
VALUES(1,1),(1,2),(1,3),(2,1);


-- ── 문서 (doc) ─────────────────────────────────────────────
INSERT INTO public.doc
    (doc_id, file_name, file_type, category, content_full, content_sum, created_at, updated_at, is_deleted, user_id)
OVERRIDING SYSTEM VALUE
VALUES
    (1, '신규_공모사업_지침.pdf', 'pdf', '공모사업',
     '가명정보 활용 데이터 분석 과제 공모 안내. 신청기간 2026-06-15~07-15, 과제당 최대 5천만원 지원. 제출서류는 사업계획서 및 개인정보 처리방침.',
     '○ 공모 대상: 가명정보 활용 데이터 분석 과제\n○ 신청 기간: 2026.06.15 ~ 07.15\n○ 지원 규모: 과제당 최대 5천만원\n○ 제출 서류: 사업계획서, 개인정보 처리방침',
     '2026-06-01 09:12:00+09', '2026-06-01 09:30:00+09', false, 2),

    (2, '감사원_처분요구서_2026.pdf', 'pdf', '감사',
     '2026년도 정기감사 결과 처분요구 사항. 개인정보 접근권한 관리 미흡 지적 및 시정 요구. 가명처리 절차 문서화 권고.',
     '○ 지적사항: 개인정보 접근권한 관리 미흡\n○ 요구사항: 권한 관리 절차 시정 및 가명처리 문서화\n○ 회신 기한: 2026.06.20',
     '2026-05-30 14:05:00+09', '2026-05-30 14:20:00+09', false, 2),

    (3, '가명정보_처리_가이드라인.pdf', 'pdf', '가이드라인',
     '개인정보 보호법 제28조의2에 따른 가명정보 처리 가이드라인. 안전성 확보조치, 결합전문기관을 통한 결합 절차, 재식별 금지 의무 등을 규정.',
     '○ 근거: 개인정보 보호법 제28조의2\n○ 핵심: 안전성 확보조치, 결합전문기관 경유\n○ 금지: 재식별 시도',
     '2026-05-28 10:40:00+09', '2026-05-28 11:00:00+09', false, 1),

    (4, '개인정보보호_내부지침_개정안.pdf', 'pdf', '기타',
     '기관 내부 개인정보 보호지침 개정안. 처리 담당자 지정, 접근권한 차등 부여, 정기 점검 주기 명시.',
     '○ 담당자 지정 및 책임 명확화\n○ 접근권한 차등 부여\n○ 분기별 정기 점검',
     '2026-05-25 16:20:00+09', '2026-05-25 16:35:00+09', false, 1);

-- ── 작업 이력 (job) — 요약/임베딩 처리 ─────────────────────
INSERT INTO public.job
    (job_id, job_start, job_finish, doc_id, user_id, job_type, job_status)
OVERRIDING SYSTEM VALUE
VALUES
    (1, '2026-06-01 09:13:00', '2026-06-01 09:14:30', 1, 2, 'summarize', 'success'),
    (2, '2026-06-01 09:15:00', '2026-06-01 09:16:10', 1, 2, 'embed',     'success'),
    (3, '2026-05-30 14:06:00', '2026-05-30 14:07:20', 2, 2, 'summarize', 'success'),
    (4, '2026-05-28 10:41:00', '2026-05-28 10:42:30', 3, 1, 'summarize', 'success'),
    (5, '2026-06-08 09:00:00', NULL,                  4, 1, 'summarize', 'running'),
    (6, '2026-06-07 18:20:00', '2026-06-07 18:20:40', 4, 1, 'embed',     'failed');

-- ── 변경 감사 로그 (history) ───────────────────────────────
INSERT INTO public.history
    (history_id, user_id, change_table, change_text, change_time)
OVERRIDING SYSTEM VALUE
VALUES
    (1, 2, 'doc',   'doc_id=1 문서 업로드 (신규_공모사업_지침.pdf)',  '2026-06-01 09:12:00+09'),
    (2, 1, 'doc',   'doc_id=3 문서 업로드 (가명정보_처리_가이드라인.pdf)', '2026-05-28 10:40:00+09'),
    (3, 1, 'draft', 'draft_id=1 기안 승인 처리',                      '2026-06-01 15:10:00+09'),
    (4, 1, 'draft', 'draft_id=3 기안 반려 처리',                      '2026-05-31 11:25:00+09'),
    (5, 1, 'users', 'user_id=2 계정 권한 변경 (실무 담당자 부여)',     '2026-05-20 09:00:00+09');

-- ── RAG 질의 로그 (rag_query) ──────────────────────────────
INSERT INTO public.rag_query
    (query_id, user_id, query_text, answer_text, source_count, created_at)
OVERRIDING SYSTEM VALUE
VALUES
    (1, 2, '가명정보 공모사업 신청 기간이 언제인가요?',
        '신규 공모사업 지침에 따르면 신청 기간은 2026년 6월 15일부터 7월 15일까지입니다.', 1, '2026-06-02 10:15:00+09'),
    (2, 2, '가명정보 처리 시 결합은 어떻게 해야 하나요?',
        '개인정보 보호법 제28조의2 및 가이드라인에 따라 결합은 지정된 결합전문기관을 통해 수행해야 하며, 재식별 시도는 금지됩니다.', 1, '2026-06-03 11:40:00+09'),
    (3, 1, '감사원 처분요구서의 회신 기한과 핵심 지적사항은?',
        '회신 기한은 2026년 6월 20일이며, 핵심 지적사항은 개인정보 접근권한 관리 미흡과 가명처리 절차 문서화 미비입니다.', 1, '2026-05-30 15:05:00+09');

-- ── RAG 출처 매핑 (rag_query_ref) ──────────────────────────
INSERT INTO public.rag_query_ref
    (ref_id, query_id, doc_id, snippet)
OVERRIDING SYSTEM VALUE
VALUES
    (1, 1, 1, '신청 기간: 2026.06.15 ~ 07.15'),
    (2, 2, 3, '결합은 결합전문기관을 통해 수행하며 재식별을 금지한다.'),
    (3, 2, 1, '제출 서류: 사업계획서, 개인정보 처리방침'),
    (4, 3, 2, '회신 기한: 2026.06.20, 접근권한 관리 미흡 지적');

-- ── 기안/결재 (draft) ──────────────────────────────────────
INSERT INTO public.draft
    (draft_id, author_id, title, content, source_doc_id, status, approver_id, reject_reason, decided_at, created_at, updated_at)
OVERRIDING SYSTEM VALUE
VALUES
    -- 승인됨
    (1, 2, '데이터 결합 가이드 검토',
        '첨부된 가명정보 처리 가이드라인을 토대로 데이터 결합 절차를 검토하였으며, 결합전문기관 경유 방식으로 진행할 것을 건의드립니다.',
        3, 'approved', 1, NULL, '2026-06-01 15:10:00+09', '2026-06-01 13:00:00+09', '2026-06-01 15:10:00+09'),

    -- 대기 (결재 정보 NULL)
    (2, 2, '가명정보 처리 승인 요청',
        '공모사업 참여를 위해 우리 기관 보유 가명정보의 활용 승인을 요청드립니다. 신청 기간은 2026.06.15~07.15이며 안전성 확보조치를 준수하겠습니다.',
        1, 'pending', NULL, NULL, NULL, '2026-06-01 14:32:00+09', '2026-06-01 14:32:00+09'),

    -- 반려됨
    (3, 2, '감사원 처분요구서 대응',
        '감사원 처분요구 사항에 대한 대응 계획을 상신합니다.',
        2, 'rejected', 1,
        '개인정보 보호 조치에 대한 구체적인 계획이 부족합니다. 가명처리 방식과 접근 권한 관리 방안을 추가로 작성해주시기 바랍니다.',
        '2026-05-31 11:25:00+09', '2026-05-30 16:00:00+09', '2026-05-31 11:25:00+09');

-- ── 알림 (notification) ────────────────────────────────────
INSERT INTO public.notification
    (noti_id, user_id, message, link, is_read, created_at)
OVERRIDING SYSTEM VALUE
VALUES
    -- 실무자(박지훈=2) 수신
    (1, 2, '''데이터 결합 가이드 검토'' 기안이 승인되었습니다', '/draft/1', false, '2026-06-01 15:10:00+09'),
    (2, 2, '''감사원 처분요구서 대응'' 반려 — 사유 확인 필요',   '/draft/3', false, '2026-05-31 11:25:00+09'),
    (3, 2, '''가명정보_처리_가이드라인'' 요약이 완료되었습니다',  '/document/3', true,  '2026-05-28 10:43:00+09'),
    -- 결재권자(김관리=1) 수신
    (4, 1, '새로운 기안 ''가명정보 처리 승인 요청''이 상신되었습니다', '/draft/2', false, '2026-06-01 14:32:00+09'),
    (5, 1, '시스템 공지: 정기 점검이 예정되어 있습니다',           NULL,        true,  '2026-06-05 09:00:00+09');

-- ── 시퀀스 재정렬 (명시적 ID 삽입 후 다음 insert 충돌 방지) ──
SELECT setval(pg_get_serial_sequence('public.doc',           'doc_id'),     (SELECT MAX(doc_id)     FROM public.doc));
SELECT setval(pg_get_serial_sequence('public.job',           'job_id'),     (SELECT MAX(job_id)     FROM public.job));
SELECT setval(pg_get_serial_sequence('public.history',       'history_id'), (SELECT MAX(history_id) FROM public.history));
SELECT setval(pg_get_serial_sequence('public.rag_query',     'query_id'),   (SELECT MAX(query_id)   FROM public.rag_query));
SELECT setval(pg_get_serial_sequence('public.rag_query_ref', 'ref_id'),     (SELECT MAX(ref_id)     FROM public.rag_query_ref));
SELECT setval(pg_get_serial_sequence('public.draft',         'draft_id'),   (SELECT MAX(draft_id)   FROM public.draft));
SELECT setval(pg_get_serial_sequence('public.notification',  'noti_id'),    (SELECT MAX(noti_id)    FROM public.notification));

COMMIT;