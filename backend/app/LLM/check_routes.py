from backend.app.LLM.main import app

print("=" * 50)
print("현재 등록된 라우트 목록:")
print("=" * 50)

for route in sorted(app.routes, key=lambda x: x.path):
    methods = route.methods if hasattr(route, 'methods') else 'N/A'
    print(f"{route.path:<30} - {methods}")

print("=" * 50)
