# 응시 카메라 + 파지 RL (시뮬레이션) 연구 계획

> 작성: 2026-11-15 · 작성자: 박준상 (SKKU) · 상태: 설계 완료, 구현 시작 전
> 이 문서는 Claude Code에서 이어서 작업하기 위한 컨텍스트 문서다.

---

## 0. 한 줄 요약

SO-101 팔 + **pan-tilt 응시 카메라 1대**를 시뮬레이션에 구성하고, **팔과 시선을 함께 제어하는 단일 RL 정책**을 학습한다.
보상은 **파지 = sparse, 시야 = dense (potential-based shaping)**. 사람 개입(HIL) 없이 **특권 상태 교사(PPO) → 시각 학생(증류)** 2단계로 학습한다.

---

## 1. 배경: 여기까지 온 과정

1. 원래 계획: SmolVLA(접근) → HIL-SERL(파지) 계층 구조로 SO-101 꽃꽂이.
2. 핸드오프 문제: RL이 넘겨받는 상태가 VLA 종료 분포와 다르면 실패. → "RL 시작 분포 = VLA 종료 분포"로 설계.
3. 선행연구 조사 결과 이 아이디어는 이미 있음 (Clegg 2018, Sequential Dexterity, RLT의 2단계 학습). 방법으로서 새롭지 않음.
4. 관점 전환: 로봇 상태가 아니라 **관측(무엇이 보이는가)** 이 핵심 → 카메라가 줄기를 응시하면 관측이 정규화됨.
5. 능동 시각 VLA는 이미 많음 (GIAVA, Eye Robot, ActiveVLA, SaPaVe, ActiveAlign). → VLA 자체보다 **응시 + 파지 RL의 보상 설계**로 좁힘.
6. 최종 결정: **HIL 없이 시뮬레이션에서만 설계·실험.**

---

## 2. 연구 질문

**메인 RQ: 명시적인 시야 보상은 눈-손 협응 학습을 돕는가, 방해하는가?**

- Eye, Robot(2025)은 시선 시연·시야 보상 없이 과제 감독만으로 눈-손 협응이 나타난다고 보고함.
- 이 연구는 dense 시야 보상의 형태(단순 합 vs potential-based)가 학습 속도·최종 성능·시선 행동에 주는 영향을 측정한다.

**보조 RQ**
- 응시 카메라가 고정 카메라보다 얇은 물체(꽃줄기) 파지에 유리한가?
- 중심와(fovea) + 주변시 입력이 전체 이미지 입력보다 학생 정책에 유리한가?
- 학습된 시선이 사람처럼 손보다 먼저 목표에 도달하는가? (인지과학 연결)

---

## 3. 시뮬레이션 환경

### 3.1 엔진
- **교사 단계(상태 기반)**: MuJoCo 또는 MJX로 충분.
- **학생 단계(이미지 대량 렌더링)**: ManiSkill3 (GPU 병렬 시뮬레이션 + 렌더링) 고려.
- 먼저 확인: 수업에서 쓴 SO101-Nexus가 어떤 엔진인지 → 붙일 수 있으면 재사용.

### 3.2 장면 구성
| 요소 | 구현 |
|---|---|
| SO-101 팔 | 공식 SO101 MJCF (`so101_new_calib.xml`). 그리퍼는 LeRobot의 0~100 매핑과 다르니 변환 직접 처리 |
| 응시 카메라 | 테이블 위 body + hinge 2개(pan, tilt) + camera. **관절 속도 제한** 필수 (실물 서보 반영) |
| 꽃줄기 | 얇은 capsule + 바닥에 스프링 ball joint (건드리면 기울어짐) |
| 홀더 | 격자 위치. 에피소드마다 위치·기울기·줄기 길이 랜덤화 |
| 카메라 파라미터 | 해상도·화각을 나중에 쓸 실물 카메라와 동일하게 |

### 3.3 행동 공간
- 팔: EE 상대 이동 (Δx, Δy, Δz, Δyaw) + 그리퍼 (또는 관절 6개)
- 시선: Δpan, Δtilt
- 총 약 7~8차원

---

## 4. 보상 설계

### 4.1 형태

```latex
r_t = \mathbb{1}[\text{grasp success}]_t + \lambda \big( \gamma \Phi(s_{t+1}) - \Phi(s_t) \big)
```

- **파지 (sparse)**: 시뮬레이터 정답으로 계산. 예: 양쪽 손가락이 줄기에 접촉 + 줄기가 2cm 이상 들림. 분류기 불필요.
- **시야 (dense)**: potential-based shaping (Ng, Harada, Russell 1999) → 최적 정책을 바꾸지 않음. "쳐다보기만 하기" 보상 해킹 방지.

### 4.2 시야 점수 Φ(s)
segmentation 렌더(또는 투영 계산)로 줄기의 화면상 정보 계산:

```latex
\Phi(s) = w_v \cdot \text{visible} - w_c \cdot \| u_{stem} - u_{center} \| - w_s \cdot | w_{stem} - w^* |
```

- visible: 줄기 픽셀이 보이는 비율
- 중심 오차: 줄기 중심의 화면 중앙으로부터 거리 (정규화)
- 크기 오차: 겉보기 폭과 목표 폭의 차이
- **가림 처리**: 그리퍼가 줄기를 가린 경우(그리퍼가 줄기 근처) 점수를 깎지 않음. 또는 접촉 단계에서 λ를 감소.

### 4.3 비교 조건 (메인 실험)
| 조건 | 보상 |
|---|---|
| A | sparse 파지만 |
| B | sparse + λ·Φ(s) (단순 합) |
| C | sparse + potential-based 시야 shaping |

예상: B는 보상 해킹, C는 학습 가속. A가 충분하면 "시야 보상 불필요"도 결과.

---

## 5. 학습 구조: 교사 → 학생

### 5.1 교사 (특권 상태, PPO, 대규모 병렬)
- 관측: 줄기 위치·기울기, 관절, 시선 각도, 줄기의 화면상 위치(투영), 이전 행동
- 알고리즘: PPO (병렬 환경 수천 개)
- sparse 보상 해결 장치:
  - **시작 위치 커리큘럼** (Reverse Curriculum 방식): 줄기 바로 옆에서 시작 → 성공률 오르면 점점 멀리
  - 선택: 스크립트 시연(IK 접근 + 노이즈)을 버퍼에 섞기 (SAC/RLPD 계열 사용 시)
- 커리큘럼 마지막 시작 분포를 나중에 "VLA 종료 분포"로 두면 VLA 연결 시 핸드오프 문제 사전 해결

### 5.2 학생 (시각, 증류)
- 관측: 응시 카메라 이미지 (중심와 crop + 저해상도 주변시) + 관절 + 시선 각도
- 학습: DAgger (학생이 움직이며 만난 상태에서 교사 행동을 정답으로)
- 선택: 마지막에 RL 미세조정
- 대안: asymmetric actor-critic (critic만 특권 상태)
- 참고: 이전 LIMO 프로젝트의 Teacher(MLP+PPO) → Student(BC+RL) 패턴과 동일

---

## 6. 실험 계획

| # | 비교 | 단계 |
|---|---|---|
| E1 | 보상 A / B / C | 교사 |
| E2 | 고정 카메라 / 응시 카메라 | 교사 + 학생 |
| E3 | 전체 이미지 / 중심와 + 주변시 | 학생 |
| E4 | 시선 행동 분석 (시선 도달 시점 vs 손 도달 시점) | 학생 |

- 시드 5개 이상
- 지표: 성공률, 학습 스텝 수, 에피소드 길이, EE jerk, 시선-손 시간차

---

## 7. 구현 순서 (체크리스트)

- [ ] 엔진 결정 (SO101-Nexus 확인 → MuJoCo or ManiSkill3)
- [ ] SO-101 MJCF 로드, 관절·그리퍼 매핑 확인
- [ ] pan-tilt 카메라 body 추가, 속도 제한, 렌더 확인
- [ ] 꽃줄기 모델 (capsule + 스프링 ball joint), 홀더 격자, 랜덤화
- [ ] segmentation 렌더 → 줄기 픽셀 마스크 → Φ(s) 계산 함수
- [ ] 스크립트 응시 제어(P 제어)로 Φ 계산이 맞는지 검증 (팔 움직여도 줄기가 중앙에 남는지)
- [ ] 파지 성공 판정 함수 (접촉 + 들림 높이)
- [ ] Gymnasium 환경 래핑 (관측: 특권 상태, 행동: 팔 + 시선)
- [ ] 보상 A/B/C 구현 (설정으로 전환)
- [ ] 시작 위치 커리큘럼
- [ ] PPO 교사 학습 (병렬 환경)
- [ ] E1 실행, 로깅 (성공률, Φ 추이, 시선 궤적)
- [ ] 학생 관측(중심와 crop) 구현 → DAgger 증류
- [ ] E2~E4

### 제안 디렉토리 구조
```
gaze_grasp/
  assets/          # SO-101 MJCF, 메시, 장면 XML
  envs/
    scene.py       # 장면 구성, 랜덤화
    gaze_grasp_env.py  # Gymnasium env
    rewards.py     # grasp_success(), view_potential(), reward A/B/C
    camera.py      # pan-tilt 제어, 렌더, segmentation, fovea crop
  teacher/         # PPO 학습
  student/         # DAgger 증류
  analysis/        # 시선-손 시간차, 플롯
  configs/
```

---

## 8. 주의사항

- 얇은 줄기의 접촉·마찰은 시뮬레이션이 부정확 → 결과는 절대 성공률보다 **조건 간 순서**로 해석.
- 카메라 관절 속도 제한 없으면 응시가 시뮬레이션에서만 잘 됨.
- 실물 이전 시: 비전 정책은 재학습 전제. 시뮬레이션은 설계·비교 실험용.
- 행동 8차원 → 탐색 어려움. 커리큘럼과 shaping이 핵심.

---

## 9. 선행 연구

### 능동 시각 / 시선
- [Look, Focus, Act (GIAVA), arXiv 2507.15833](https://arxiv.org/html/2507.15833v2) — 시선을 행동 공간에 포함 + 중심와 ViT, flow matching 정책
- [Eye, Robot, arXiv 2506.10968](https://arxiv.org/html/2506.10968v2) — 기계 안구 시선을 RL로 학습, 시선 시연 없이 눈-손 협응 출현. **메인 비교 대상**
- [Active Vision Might Be All You Need (AV-ALOHA), arXiv 2409.17435](https://arxiv.org/html/2409.17435) — 카메라 전용 7자유도 팔
- [Look Before You Act (ActiveAlign), arXiv 2610.23378](https://arxiv.org/abs/2610.23378) — 카메라 재배치가 VLA 초기 분포를 깨뜨림 (π0.5 성공률 46.3% → 11.5%)
- [ActiveVLA, arXiv 2601.08325](https://arxiv.org/pdf/2601.08325) — VLA에 능동 시점 선택 + 3D 줌인
- [SaPaVe, CVPR 2026](https://openaccess.thecvf.com/content/CVPR2026/papers/Liu_SaPaVe_Towards_Active_Perception_and_Manipulation_in_Vision-Language_Action_Models_CVPR_2026_paper.pdf) — 카메라·조작 행동 공간 분리
- [Gaze-based dual resolution IL, arXiv 2102.01295](https://ar5iv.arxiv.org/html/2102.01295) — 중심와/주변시 이중 해상도, 바늘 꿰기
- [Gaze-Regularized VLA, arXiv 2603.23202](https://www.emergentmind.com/papers/2603.23202), [GazeVLA, arXiv 2604.22615](https://www.emergentmind.com/papers/2604.22615)

### 시작 분포 / 핸드오프 (배경)
- [Reverse Curriculum Generation (CoRL 2017)](https://arxiv.org/pdf/1707.05300v3.pdf) — 시작 분포 커리큘럼
- [Sequential Dexterity (CoRL 2023)](https://arxiv.org/abs/2309.00987), [T-STAR (CoRL 2021)](https://arxiv.org/abs/2111.07999), [SCaR (NeurIPS 2024)](https://papers.nips.cc/paper_files/paper/2024/file/ca92ff06d973ece92cecc561757d500e-Paper-Conference.pdf)
- [RouteRLT, arXiv 2609.26467](https://arxiv.org/html/2609.26467v1), [RL Token (RLT), arXiv 2604.23073](https://arxiv.org/html/2604.23073v1)

### 보상 shaping
- Ng, Harada, Russell (1999), "Policy invariance under reward transformations" — potential-based shaping 이론

### 도구
- [LeRobot gym_hil (MuJoCo, 참고용)](https://huggingface.co/docs/lerobot/en/hilserl_sim)
- SO101 URDF/MJCF: SO-ARM100 저장소의 `so101_new_calib.xml`
- ManiSkill3 (GPU 병렬 시뮬레이션 + 렌더링)

---

## 10. 이후 확장 (선택)

- 학생 정책 앞에 SmolVLA 접근 단계를 붙이고, 커리큘럼 마지막 시작 분포를 VLA 종료 분포로 설정 → 핸드오프 연구와 연결
- 실물: STS3215 서보 2개로 pan-tilt 제작, 시뮬레이션에서 정한 카메라 위치 그대로
