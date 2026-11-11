"""Generate an explicitly synthetic Vietnamese retrieval collection."""
import json
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parent
INTENTS = {
    "return": ("Đổi trả", "Khách có thể trả lại {sku} trong {days} ngày khi còn nguyên tem.",
               ["Tôi muốn gửi lại {sku}, thời hạn là bao lâu?", "{sku} được đổi trả trong mấy ngày?"]),
    "warranty": ("Bảo hành", "{sku} được sửa chữa miễn phí do lỗi nhà sản xuất trong {months} tháng.",
                 ["{sku} hỏng do nhà sản xuất thì sửa miễn phí bao lâu?", "Thời gian bảo hành của {sku} là bao nhiêu?"]),
    "shipping": ("Vận chuyển", "Giao {sku} mất {ship} ngày làm việc. Đơn từ {threshold} đồng được miễn cước.",
                 ["Đặt {sku} thì bao lâu nhận được hàng?", "Mua {sku} cần bao nhiêu tiền để miễn phí giao hàng?"]),
    "refund": ("Hoàn tiền", "Sau khi duyệt, tiền mua {sku} được chuyển lại trong {refund} ngày qua phương thức gốc.",
               ["Sau khi duyệt trả {sku}, khi nào tiền về tài khoản?", "Hoàn tiền cho {sku} mất mấy ngày?"]),
    "cancel": ("Hủy đơn", "Có thể ngừng đơn {sku} trước lúc bàn giao cho bên vận chuyển. Đơn đã giao không thể hủy.",
               ["Làm sao dừng đơn mua {sku} trước khi gửi đi?", "Đã giao cho vận chuyển thì có hủy đơn {sku} được không?"]),
    "payment": ("Thanh toán", "Mua {sku} có thể trả bằng thẻ, chuyển khoản hoặc tiền mặt khi nhận hàng.",
                ["Tôi có thể trả tiền mặt lúc nhận {sku} không?", "Những phương thức thanh toán khi mua {sku} là gì?"]),
}


def build(seed=42):
    rng = random.Random(seed)
    groups = [f"SP{i:03d}" for i in range(1, 31)]
    rng.shuffle(groups)
    membership = {g: ("train" if i < 18 else "dev" if i < 24 else "test")
                  for i, g in enumerate(groups)}
    docs, queries = [], []
    for sku in sorted(groups):
        values = dict(sku=sku, days=rng.choice([7, 10, 14]), months=rng.choice([6, 12, 24]),
                      ship=rng.choice([2, 3, 5]), threshold=rng.choice([300000, 500000, 700000]),
                      refund=rng.choice([3, 5, 7]))
        for intent, (title, body, templates) in INTENTS.items():
            doc_id = f"{sku}-{intent}"
            docs.append({"id": doc_id, "group": sku, "title": f"{title} — {sku}",
                         "text": body.format(**values)})
            for j, template in enumerate(templates):
                queries.append({"id": f"q-{doc_id}-{j}", "group": sku,
                                "split": membership[sku], "text": template.format(**values),
                                "qrels": {doc_id: 1}})
    return {"version": "synthetic-v1", "synthetic": True, "seed": seed,
            "description": "30 synthetic products; shared templates; split by product, not by template.",
            "documents": docs, "queries": queries}


if __name__ == "__main__":
    path = ROOT / "data" / "collection.json"
    path.parent.mkdir(exist_ok=True)
    path.write_text(json.dumps(build(), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Saved {path}: 180 documents, 360 queries (216 train / 72 dev / 72 test).")
