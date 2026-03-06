## 贡献流程
1. Fork 仓库
2. 创建分支：git checkout -b feat/your-feature-name
3. 提交更改：git commit -m "feat: add new feature"
4. 推送到 Fork：git push origin feat/your-feature-name
5. 创建 Pull Request 到主仓库的 main 分支

## Commit Message 规范
我们使用 Conventional Commits：
```
<type>[optional scope]: <description>

[optional body]

[optional footer(s)]
```

类型说明：
- feat/ - 新功能
- fix/ - Bug 修复
- docs/ - 文档更新
- refactor/ - 代码重构
- test/ - 测试相关
- chore/ - 杂项（依赖更新等）

示例：
```
feat(auth): add JWT token validation

- Implement token verification
- Add token refresh endpoint

Closes #123
```

## Code Review 流程
1. 所有 PR 必须至少 1 个 approve 才能合并
2. CI 必须通过
3. 解决所有评论后才能合并
4. 使用 "Squash and merge" 合并

## 发布流程
只有 maintainer 可以发布新版本：
```
make release VERSION=1.0.0
```
