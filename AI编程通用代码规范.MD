# AI 编程通用代码规范（前后端通用版）

## 0. 架构规范（优先级最高 / P0）

> **本章节为全规范最高优先级。**
> 当本规范其他条款与架构规范发生冲突时，**以本章节为准**。

1. **必须采用明确架构模式（MVC 或三层架构）**
   - MVC：`Controller` 负责请求分发，`Model` 负责领域与数据，`View` 负责展示。
   - 三层：`Controller(接口层)` → `Service(业务层)` → `Repository/DAO(数据层)`。

2. **严格分层，禁止越层访问**
   - 禁止 `Controller -> Repository` 直连，必须经 `Service`。
   - 禁止 `View` 承载业务逻辑与数据访问。
   - 禁止在前端文件中直接实现后端数据库逻辑。

3. **模块边界清晰，按业务域组织目录（优先）**
   - 推荐以业务域组织：如 `modules/user`、`modules/order`。
   - 每个业务域内再分层：`controller|service|repository|domain|dto`。
   - 业务代码应高内聚、低耦合，跨模块调用必须通过明确接口。

4. **严禁跨模块私有实现直连**
   - 禁止模块 A 直接读写模块 B 私有数据表/私有对象。
   - 必须通过模块 B 对外公开的服务接口交互。

5. **前后端不分离项目的边界要求**
   - 可使用模板引擎渲染（如 Jinja2），但仍需保持职责边界：
     - 控制层：路由与请求编排
     - 业务层：业务规则与校验
     - 数据层：持久化与查询
     - 模板层：页面展示

---

## 1. 总体原则

1. **可读性优先于“炫技”**
   - 代码首先给人读，其次给机器跑。
2. **一致性高于个人偏好**
   - 同项目内风格必须统一。
3. **最小惊讶原则**
   - 命名、结构、行为应符合团队直觉。
4. **可维护性优先**
   - 避免过度抽象和过早优化。
5. **变更可追溯**
   - 代码、注释、提交信息保持一致。

---

## 2. 代码格式与缩进

1. **缩进统一**
   - 默认推荐：**4 空格**（Python 等）或 **2 空格**（JS/TS 前端常见）。
   - 一个项目只能选一种，禁止混用 Tab 与空格。
2. **行宽限制**
   - 建议每行不超过 **100~120** 字符。
3. **空行规则**
   - 逻辑块之间保留空行，提升可读性。
   - 函数与函数之间空一行或按语言约定处理。
4. **花括号/语句块**
   - 即使单行分支，也建议使用完整语句块，减少未来改动风险。
5. **自动格式化**
   - 必须使用统一格式化工具（如 Prettier/Black/gofmt/rustfmt 等），禁止手工风格冲突。

---

## 3. 命名规范

> 核心原则：**见名知意、统一语义、避免歧义**。

1. **变量名**
   - 使用有业务含义的词，避免 `tmp`, `data`, `obj` 泛化命名（循环临时变量除外）。
2. **函数/方法名**
   - 使用动词短语，体现行为：`getUserById`, `calculateTotal`, `sendEmail`。
3. **类/类型名**
   - 使用名词：`UserService`, `OrderRepository`, `PaymentResult`。
4. **常量名**
   - 全大写下划线（按语言习惯）：`MAX_RETRY_COUNT`。
5. **布尔命名**
   - 建议 `is/has/can/should` 前缀：`isActive`, `hasPermission`。
6. **集合命名**
   - 使用复数或语义后缀：`users`, `orderList`, `userMap`。
7. **避免缩写**
   - 除公认缩写外（ID、URL、HTTP），尽量写全。
8. **语言习惯优先**
   - Java/C#: `camelCase` + `PascalCase`
   - Python: `snake_case`
   - 常量按语言社区规范执行。

---

## 4. 注释规范

1. **注释解释“为什么”，不是“做了什么”**
   - 代码能表达“做什么”，注释应补充背景、约束、边界、风险。
2. **对外接口必须有文档注释**
   - 函数用途、参数、返回值、异常、副作用、示例（按语言文档系统）。
3. **复杂逻辑必须注释**
   - 算法、兼容性、性能折中、业务规则。
4. **禁止无效注释**
   - 过时注释必须同步更新或删除。
5. **TODO/FIXME 规范**
   - 必须带责任人或 issue 编号，避免永久遗留。

---

## 5. 函数与模块设计

1. **单一职责**
   - 一个函数只做一件事。
2. **函数长度控制**
   - 优先保持短小（建议 20~50 行以内，视语言和业务可放宽）。
3. **参数控制**
   - 参数过多（>4）时，使用对象/结构体封装。
4. **避免深层嵌套**
   - 优先早返回（guard clause）。
5. **模块边界清晰**
   - 控制层、服务层、数据层职责明确，不互相越界。
6. **按业务模块划分（优先）**
   - 以业务域组织目录（如 `user/`, `order/`, `payment/`），每个业务域内再分层，而不是全局按技术类型散落。
   - 业务模块应具备高内聚、低耦合，跨模块调用必须通过明确接口。
7. **前后端代码严格分离**
   - 前端页面/组件/状态管理代码与后端控制器/服务/数据访问代码不得放在同一文件。
   - 禁止在前端文件中直接写后端数据库访问逻辑；禁止在后端文件中嵌入前端渲染实现细节（模板引擎场景除外且需边界清晰）。
8. **架构模式落地（MVC 或三层架构）**
   - MVC 参考：`Controller` 负责请求分发，`Model` 负责领域与数据，`View` 负责展示。
   - 三层架构参考：`Controller(接口层)` → `Service(业务层)` → `Repository/DAO(数据层)`。
   - 任何一层不得越层访问（如 Controller 直连数据库）。

---

## 6. 错误处理与日志

1. **禁止吞异常**
   - 必须显式处理、包装或上抛。
2. **错误信息可诊断**
   - 包含关键上下文（请求ID、用户ID、资源ID）。
3. **日志分级**
   - `DEBUG/INFO/WARN/ERROR` 语义一致。
4. **日志不泄露敏感信息**
   - 不打印密码、密钥、token、隐私数据。
5. **统一错误码/错误结构**
   - 前后端接口返回需稳定且可机器解析。

---

## 7. 安全规范（通用）

1. **输入校验**
   - 所有外部输入均视为不可信。
2. **防注入**
   - SQL/命令/模板注入必须使用参数化、转义和白名单。
3. **鉴权与权限控制**
   - 认证（你是谁）与授权（你能做什么）分离。
4. **敏感配置外置**
   - 密钥放环境变量或密钥管理系统，禁止硬编码。
5. **依赖安全**
   - 定期升级依赖，修复高危漏洞。

---

## 8. 前端专项补充

1. **组件设计**
   - 组件职责单一，避免“巨型组件”。
2. **状态管理**
   - 区分本地状态与全局状态，避免状态散乱。
3. **可访问性（a11y）**
   - 语义化标签、键盘可用、合理 aria 属性。
4. **性能**
   - 减少不必要渲染、按需加载、资源压缩与缓存。
5. **API 调用**
   - 统一封装请求层，统一错误处理和重试策略。
6. **按业务模块组织前端目录**
   - 建议结构：`features/<biz>/components|pages|hooks|services|types`。
   - 公共能力放 `shared/` 或 `common/`，禁止把业务逻辑堆在 `utils` 大杂烩目录。

---

## 9. 后端专项补充

1. **接口设计**
   - URL、方法、状态码、错误结构一致。
2. **幂等性**
   - 写接口在重试场景应尽可能幂等。
3. **事务与一致性**
   - 明确事务边界，避免隐式跨层事务。
4. **并发控制**
   - 注意锁、竞争条件、重复提交。
5. **可观测性**
   - 必须有日志、指标、追踪（Tracing）能力。
6. **按业务模块组织后端目录**
   - 建议结构：`modules/<biz>/controller|service|repository|domain|dto`（可按语言调整）。
   - 公共中间件、基础设施能力放 `infrastructure/` 或 `shared/`，避免业务代码反向依赖基础设施细节。
7. **严禁跨层/跨模块直连**
   - 禁止 `Controller -> Repository` 直连，必须经 `Service`。
   - 禁止模块 A 直接读写模块 B 私有数据表/私有对象，需通过模块公开服务接口。

---

## 10. 测试规范

1. **可测试性优先**
   - 业务逻辑与外部依赖解耦。
2. **测试分层**
   - 单元测试、集成测试、端到端测试职责清晰。
3. **关键路径必测**
   - 正常流程、边界条件、异常流程。
4. **测试命名**
   - 体现场景与预期：`should_return_401_when_token_invalid`。
5. **CI 门禁**
   - 测试失败禁止合并。

---

## 11. Git 与代码评审

1. **提交粒度小而清晰**
   - 一次提交解决一个问题。
2. **提交信息规范**
   - 建议：`type(scope): summary`（如 `fix(auth): handle token expiry`）。
3. **PR 描述完整**
   - 背景、改动点、影响面、测试说明。
4. **Code Review 关注点**
   - 正确性、可维护性、安全、性能、一致性。
5. **禁止直接推主干**
   - 走分支 + Review 流程。

---

## 12. AI 生成代码专项要求（重点）

1. **AI 代码必须人工复核**
   - 不可直接“盲合并”。
2. **先有约束再生成**
   - 明确语言版本、框架版本、风格与边界条件。
3. **生成后必须补齐**
   - 错误处理、日志、测试、注释、类型声明（若适用）。
4. **验证依赖与许可证**
   - 避免引入不合规或高风险依赖。
5. **保留决策记录**
   - 对关键改动写明“为何如此实现”。

---

## 13. 可执行目录模板示例（用于快速落地）

> 说明：以下模板用于规范落地参考，团队可按业务规模裁剪；但**分层边界与调用方向不可破坏**。

### 13.1 Python Flask（前后端不分离，MVC/三层混合落地）

```text
project/
  app/
    __init__.py                 # create_app、注册蓝图、中间件
    config.py                   # 配置项（环境变量、常量）
    controllers/                # Controller：路由与请求编排
      __init__.py
      todo_controller.py
    services/                   # Service：业务规则、参数校验、事务边界
      __init__.py
      todo_service.py
    repositories/               # Repository：数据库读写
      __init__.py
      todo_repository.py
    templates/                  # View：Jinja 模板
      todo_list.html
    static/                     # 静态资源（css/js/img）
      css/
      js/
  tests/
    test_todo_service.py
    test_todo_controller.py
  run.py                        # 启动入口
  requirements.txt
```

调用方向：`Controller -> Service -> Repository`，模板层只做展示。

### 13.2 Java Spring Boot（三层架构）

```text
src/main/java/com/example/project/
  modules/
    user/
      controller/
        UserController.java
      service/
        UserService.java
        impl/UserServiceImpl.java
      repository/
        UserRepository.java
      domain/
        User.java
      dto/
        UserCreateRequest.java
        UserResponse.java
    order/
      controller/
      service/
      repository/
      domain/
      dto/
  infrastructure/
    config/
    exception/
    logging/
src/main/resources/
  application.yml
src/test/java/com/example/project/
```

约束：禁止 `controller` 直接调用 `repository`；跨模块通过 service 接口。

### 13.3 Node.js NestJS（模块化三层）

```text
src/
  modules/
    user/
      user.controller.ts        # Controller
      user.service.ts           # Service
      user.repository.ts        # Repository（或 data-access provider）
      dto/
        create-user.dto.ts
      entities/
        user.entity.ts
      user.module.ts
    order/
      order.controller.ts
      order.service.ts
      order.repository.ts
      dto/
      entities/
      order.module.ts
  shared/
    filters/
    guards/
    interceptors/
    pipes/
  main.ts
test/
```

约束：控制器只处理 HTTP 语义；业务逻辑进入 service；数据访问进入 repository/provider。

---

## 可直接落地的最小执行清单（推荐）

- [ ] 架构评审门禁：PR 必须说明分层边界与模块归属
- [ ] CI 增加架构检查（禁止越层依赖）
- [ ] 统一格式化工具 + lint 规则
- [ ] 统一命名规范文档
- [ ] 必填 PR 模板（背景/改动/测试/风险）
- [ ] CI：lint + test + build 必过
- [ ] 敏感信息扫描与依赖漏洞扫描
- [ ] AI 代码复核清单（安全/异常/测试/性能）
