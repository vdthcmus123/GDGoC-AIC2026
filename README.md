# Bomberland GDGoC AI Challenge

Repository này chứa mã nguồn của đội thi DipperxBill, kết thúc cuộc thi với **rank #4** trên final leaderboard.

Bomberland là game chiến thuật theo lượt lấy cảm hứng từ Bomberman trên map
13x13 (cuộc thi mô phỏng trò chơi tuổi thơ Bom IT). Mỗi agent phải chọn action trong time budget rất chặt, đồng thời cân
bằng nhiều mục tiêu: sống sót, phá box, nhặt item, đặt bom an toàn, bẫy đối
thủ, tạo late-game pressure và tối ưu tie-break.

## Kết Quả

- Final leaderboard rank: **#4**
- Agent type: hybrid neural + rule-based + beam-search planner
- Runtime: không quá 100ms/step

## Cấu Trúc Repository

```text
.
|-- README.md
|-- requirements.txt
|-- submission/
|   |-- agent.py
|   `-- policy.pt
`-- notebooks/
    `-- notebook.ipynb
```

## Kiến Trúc Agent

Agent cuối cùng là một hệ thống hybrid gồm ba lớp chính.

### 1. Rule-Based Safety Core

Safety core mô phỏng:

- blast propagation;
- bomb timer và chain reaction;
- future danger planes;
- enemy bomb threats;
- safe action mask;
- escape routes sau khi đặt bom.

Một luật quan trọng là **two-exit bomb rule**: agent chỉ tự nguyện đặt bom
khi simulator tìm thấy ít nhất hai đường thoát an toàn. Luật này giúp giảm
self-trap và hạn chế rank-3 finishes.

### 2. Neural Policy

Neural policy là phần học được từ dữ liệu và self-play. Có thể hiểu đơn giản:
rule-based safety core trả lời câu hỏi **"action nào có thể sống được?"**,
còn neural policy trả lời câu hỏi **"trong các action hợp lệ, action nào
giống hành vi của một agent mạnh nhất?"**.

Model được thiết kế dạng **CNN-GRU actor-critic** và export bằng
**TorchScript** để chạy nhanh trong official evaluator.

Input của model gồm hai nhóm:

- **Board channels 13x13**: map, wall, box, item, vị trí bản thân, vị trí
  đối thủ, bomb, bomb timer, bomb radius và future danger planes.
- **Scalar features**: số bom còn lại, bomb radius, số đối thủ còn sống,
  game phase, khoảng cách tới enemy gần nhất, số safe moves và số box có thể
  phá nếu đặt bom.

Bên trong model:

- **CNN** đọc cấu trúc không gian của board, ví dụ corridor, box cluster,
  bomb line và vị trí enemy.
- **GRU** giữ short-term memory để agent hiểu nhịp hành động gần đây, đặc
  biệt là sau khi đặt bom, khi chạy khỏi blast, hoặc khi lặp lại late-game
  pressure.
- **Actor head** sinh `action logits` cho 6 actions: đứng yên, đi 4 hướng,
  và đặt bom.
- **Critic head** ước lượng state value để hỗ trợ PPO training.

Điểm quan trọng: neural policy **không được quyền quyết định một mình**.
Logits của model luôn đi qua legal/safety mask và beam-search planner. Nhờ
vậy model học được intuition chiến thuật từ dữ liệu, còn rule/planner vẫn
giữ vai trò chặn các action tự sát.

### 3. Beam-Search Planner

Beam-search planner là lớp ra quyết định cuối cùng trong runtime. Planner
nhận `action logits` từ neural policy, nhưng không chọn ngay action có xác
suất cao nhất. Thay vào đó, nó dùng logits như **priors**, rồi mô phỏng ngắn
một số tương lai khả thi để chọn action an toàn và có giá trị chiến thuật cao.

Luồng quyết định của planner:

1. Lấy safe action mask từ safety core.
2. Chuyển neural logits thành prior score cho từng action hợp lệ.
3. Với mỗi action đầu tiên, mô phỏng vị trí tương lai trong vài bước.
4. Loại các nhánh đi vào danger plane hoặc bị kẹt bởi bomb/wall/box.
5. Chấm điểm nhánh còn lại bằng tactical score.
6. Trả về action đầu tiên của nhánh có tổng điểm cao nhất.

Tactical score gồm nhiều tín hiệu:

- **survival margin**: còn bao nhiêu tick an toàn trước blast;
- **mobility**: sau action đó còn bao nhiêu ô/đường di chuyển;
- **box/item objective**: có phá box, mở đường hoặc nhặt item được không;
- **post-bomb escape**: nếu vừa đặt bom, action có giúp rời blast line không;
- **enemy pressure**: bomb có ép enemy vào vùng ít đường thoát hơn không;
- **trap potential**: bomb có giảm escape region của enemy không;
- **late-game tie-break activity**: về cuối trận, ưu tiên bomb pressure an
  toàn để tăng cơ hội thắng tie-break.

Planner là lý do agent không chỉ là một neural network thuần. Neural policy
giúp agent có phong cách giống top agents, còn planner biến phong cách đó
thành action cụ thể an toàn dưới ràng buộc 100 ms/step. Nhờ vậy agent có thể
farm và snowball sớm, sau đó aggressive hơn ở late game mà vẫn hạn chế
self-trap.

## Training Pipeline

Pipeline training được xây theo nhiều giai đoạn:

1. **Rule-based warm start** từ tactical farming policy.
2. **Behavior cloning** từ handcrafted agents để học movement, bombing và
   escape rhythm cơ bản.
3. **Curriculum learning**: bắt đầu từ survival/farming, sau đó tăng dần độ
   khó với stronger baselines.
4. **PPO-style self-play** để policy học tương tác nhiều agent cùng lúc.
5. **Elite-log recurrent imitation** từ các top leaderboard agents để học
   chiến thuật theo phase trận.
6. **PFSP (Prioritized Fictitious Self-Play) league training** để sample opponent snapshots/clones theo độ khó.
7. **Conservative checkpoint selection** dựa trên average rank, rank-0 rate,
   rank-3 rate, draw rate, bomb rate và wait rate.


## Strategy Summary

Chiến thuật cuối cùng theo hướng snowball:

- **Opening:** phá box và nhặt item an toàn để tăng bomb count/radius.
- **Mid game:** dùng lợi thế item để kiểm soát corridor và ép vị trí đối thủ.
- **Late game:** đặt bom liên tục nhưng có kiểm soát để tăng pressure và
  tie-break metrics, trong khi vẫn ưu tiên sống đến cuối.

## Cách Chạy Submission

Submission chính nằm trong:

```text
submission/agent.py
submission/policy.pt
```

`agent.py` cần `policy.pt` ở cùng thư mục. Official evaluator sẽ khởi tạo:

```python
agent = Agent(agent_id)
action = agent.act(obs)
```

Cài dependencies tối thiểu:

```bash
pip install -r requirements.txt
```

## Notebook

`notebooks/notebook.ipynb` là bản public của
notebook training (không bao gồm dataset training).

## Ghi Chú

Raw match logs, Kaggle artifacts và intermediate checkpoints không được đưa
vào repository vì kích thước lớn. Repository này tập trung vào final
deployable agent và notebook giải thích pipeline.
