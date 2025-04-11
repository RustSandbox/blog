

## A Coder's Guide to the Official Rust MCP Toolkit (`rmcp`)

Hey there! Ever wondered how super-smart computer programs, like the AI assistants you might chat with, can use other tools or get information from different places? Like, how does an AI know the weather, or how can it use a calculator? That's where something called the **Model Context Protocol (MCP)** comes in. Let's break it down!

**I. What is this MCP Thing?**

Imagine you have a super-smart AI, called a **Large Language Model (LLM)** – think of things like ChatGPT or Claude. These AIs are great at understanding and generating text, but they often live inside their own "digital brain."

Now, imagine you want this AI to do something specific, like look up the latest score for your favorite sports team or use a special calculator you built. The AI needs a way to talk to the *outside world* – other computer programs, databases (where data is stored), or tools.

**MCP is like a universal translator and messenger service for AIs.**

* It sets up **standard rules** (a "protocol") for how the AI application and these external systems should talk to each other.
* It uses a common format, often something called **JSON-RPC** (think of it like a very specific way of writing text messages so computers can easily understand them).
* This lets the AI say, "Hey, I need to use that calculator tool!" or "Can you fetch me the latest news?"
* The external system can then understand the request, do the job, and send back the answer in a structured way the AI understands.

**Why is this cool?**

Instead of just giving the AI a long text instruction, MCP lets developers build apps where the AI can clearly ask for specific tools or data. It helps keep track of the conversation ("context") so the AI remembers what's going on.

MCP helps separate jobs: one part provides the data or tools (the "server"), and another part connects to the AI and manages the conversation (the "client").

**What about Rust?**

**Rust** is a popular programming language known for being fast and safe (it helps prevent common coding errors). To use MCP in Rust, developers need tools to make it easier. That's where an **SDK** comes in.

An **SDK (Software Development Kit)** is like a **Lego kit** for programmers. It gives you pre-built pieces and instructions to build something specific – in this case, applications that use MCP.

This guide is about the **official Rust SDK for MCP**, which is often called `rmcp`. It's the main, supported toolkit for using MCP with Rust.

**II. Different MCP Toolkits in Rust**

When you look for MCP tools in Rust, you might find a few with similar names. It's like finding different brands of Lego kits for building a spaceship – they might do similar things, but the official one is often the most reliable.

* **A. The Official One: `rmcp`**
    * **This is the main toolkit we'll focus on.** It's kept up-to-date by the people who manage MCP.
    * It uses something called **Tokio**, which is a popular Rust tool for handling many tasks at once without getting stuck waiting (we call this **asynchronous** or **async** programming – like a chef juggling multiple cooking steps).
    * It lets you build both the **client** (the part asking for things) and the **server** (the part providing tools/data).
    * It supports different ways for the client and server to talk (like over the internet, or between programs on the same computer – these are called **transport layers**).
    * It uses **traits** – think of these like defining a set of "skills" or "abilities" a piece of code *must* have to work as a client or server.
    * It has **macros** – these are like **magic spells** or **shortcuts** in Rust code (they often start with `#[...]` or end with `!`). They help automatically write some of the tricky code needed, especially for setting up tools on the server.

* **B. Just the Blueprints: `rust-mcp-schema`**
    * This toolkit is different. It *only* provides the **definitions** or **blueprints** for the messages MCP uses. Think of it as having the design drawings for the Lego bricks, but not the bricks themselves or the instructions on how to connect them.
    * It helps make sure the messages are structured correctly using Rust's data types (`struct`s and `enum`s).
    * Other SDKs (like `rmcp`) might use this blueprint kit inside them.

* **C. Other Toolkits (Briefly!)**
    * There are other MCP toolkits for Rust made by the community (other developers). They might have different features or be works-in-progress.
    * Examples: `mcp_rust_sdk`, `rust-mcp-sdk`, `mcp-sdk-rs`, `MCPR`, `mcp_rs`, `mcp-sdk`.
    * It's good to know they exist, but for learning the standard way, **`rmcp` is the one to focus on.**

**Comparison Table (Simplified):**

| Toolkit Name      | Official? | What it Does                                  | Key Features                                           | Status            |
| :---------------- | :-------- | :-------------------------------------------- | :----------------------------------------------------- | :---------------- |
| **`rmcp`** | **Yes** | Full Toolkit (Client & Server)                | Official, Async (Tokio), Handles communication, Macros | Actively Updated  |
| `rust-mcp-schema` | No        | Only Message Blueprints                       | Just defines message structures                        | Stable (for refs) |
| Others...         | No        | Often Full Toolkits (Client & Server attempt) | Various features, often community-made               | Varies            |

**III. Getting Started with `rmcp`**

Ready to see how you might use this `rmcp` toolkit in a Rust project?

**A. Installation (Adding the Toolkit to Your Project)**

In Rust, you manage your project's "ingredients" (external code libraries, called **crates**) using a file called `Cargo.toml`. To use `rmcp`, you add it to this file.

```toml
# This goes inside your Cargo.toml file, under [dependencies]

[dependencies]
# Add rmcp, specifying the version and the 'features' you need
rmcp = { version = "0.1", features = ["server", "client", "transport-io", "transport-sse", "macros"] }

# You'll almost always need Tokio for async stuff
tokio = { version = "1", features = ["full"] }

# These help handle data structures and convert them to/from text (like JSON)
serde = { version = "1.0", features = ["derive"] }
serde_json = "1.0"

# This helps rmcp's macros generate code related to data structures
schemars = "0.8"

# A common helper for managing errors in Rust
anyhow = "1.0"
```

* **`[dependencies]`**: This section lists the external crates your project needs.
* **`rmcp = { ... }`**: Tells Rust to include the `rmcp` crate.
* **`version = "0.1"`**: Specifies which version of `rmcp` to use (check for the latest!).
* **`features = [...]`**: This is important! It's like choosing **optional add-ons** for the toolkit.
    * `"server"`: Need this if you're building an MCP server.
    * `"client"`: Need this if you're building an MCP client.
    * `"macros"`: Enables those helpful code-writing macros (`#[tool]`, `tool_box!`).
    * `"transport-io"`: Enables talking using standard input/output (like keyboard and screen).
    * `"transport-sse"`: Enables talking using Server-Sent Events (a web technology).
    * `"transport-child-process"`: Enables talking to a program launched as a separate process.
* **`tokio`**: The async runtime (the engine for handling many tasks at once). `features = ["full"]` enables all of Tokio's helpers.
* **`serde`, `serde_json`**: Libraries for **serialization** (turning Rust data like `struct`s into text like JSON) and **deserialization** (turning text back into Rust data). Super common and useful.
* **`schemars`**: Helps automatically describe the structure (schema) of your data, which MCP tools need.
* **`anyhow`**: A popular crate for making error handling easier in Rust applications.

**B. SDK Structure (How the Toolkit is Organized)**

Inside the `rmcp` crate, the code is organized into different modules (like folders for code) based on what they do:

* `handler`: Defines the core "skill sets" (**traits**) needed for servers (`ServerHandler`) and clients (`ClientHandler`). This is where you'll write most of your app's specific logic.
* `transport`: Deals with *how* the client and server talk to each other (the communication channels). Contains the `IntoTransport` trait and specific ways like `StdioServerTransport` (for keyboard/screen talking).
* `model`: Defines the Rust **`struct`s** (custom data containers) and **`enum`s** (multiple-choice types) that represent all the different MCP messages (requests, responses, errors, etc.).
* `service`: Basic building blocks related to running the MCP service (client or server).
* `error`: Defines the main error type (`McpError`) used by the `rmcp` toolkit when things go wrong.
* `macros`: (Used via things like `rmcp::tool`) Provides those helpful code-writing macros.

**C. Core `rmcp` Traits (The "Skill Sets")**

Traits are a key concept in Rust. Think of them as defining a **contract** or a **set of abilities** that a type (like a `struct`) must have. `rmcp` uses three main ones:

1.  **`IntoTransport`**:
    * **Purpose:** This trait is like a **universal adapter** for communication. It allows `rmcp` to work with different ways of sending and receiving messages (like talking over the internet, through files on the computer, etc.) without the main code needing to know the specific details.
    * **How it works:** Anything that has the `IntoTransport` skill must be able to be split into two parts: a **`Sink`** (for *sending* messages out) and a **`Stream`** (for *receiving* messages in).
    * **Flexibility:** This means you can use standard input/output, network connections (like TCP), or even custom communication methods, as long as they can provide a `Sink` and a `Stream`.

2.  **`ServerHandler`**:
    * **Purpose:** This defines the "skill set" required to **act as an MCP server**. You'll create your own `struct` (to hold the server's data or state) and then implement this trait for it.
    * **Key Skills (Methods):**
        * `get_info()`: Provides basic info about the server (name, version).
        * `handle_request()`: This is the main brain! It receives requests from the client (like "list tools" or "call a specific tool") and needs to figure out how to respond. It runs **asynchronously** (using `async fn`).
        * `handle_notification()`: Handles simple messages (notifications) from the client that don't need a direct response.

3.  **`ClientHandler`**:
    * **Purpose:** This defines the "skill set" required to **act as an MCP client**. It's mainly about reacting to messages *from* the server.
    * **Key Skills (Methods):**
        * `get_info()`: Provides basic info about the client.
        * `handle_request()`: Handles requests *from* the server (less common, but possible).
        * `handle_notification()`: Reacts to notifications from the server (like "progress update" or "a resource you care about has changed").
        * `get_peer()` / `set_peer()`: Manages the connection object (`Peer`).
    * **Simple Clients:** If your client just sends requests and doesn't need to react much to the server, you can often use a very simple handler, like the empty tuple `()`!

**IV. Understanding `rmcp` Transports (How They Talk)**

**Transports** are the actual communication pipes between the MCP client and server. `rmcp`'s `IntoTransport` trait makes it easy to swap these pipes.

**A. The `IntoTransport` Adapter Revisited**

Remember, this trait is the key. It separates the "what" (MCP logic) from the "how" (communication channel). As long as something can give you a way to send (`Sink`) and receive (`Stream`) MCP messages asynchronously, it can be used as a transport.

**B. Available Communication Pipes (Transports)**

`rmcp` comes with several built-in transport options (you enable them with `features` in `Cargo.toml`):

| Transport Name         | `rmcp` Feature Flag         | How it Communicates        | Typical Use Case                                     |
| :--------------------- | :-------------------------- | :------------------------- | :--------------------------------------------------- |
| **Stdio** | `transport-io`              | Standard Input/Output      | Simple tools on your computer, one program talking to another it launched |
| **SSE Server** | `transport-sse-server`      | Server-Sent Events (Web)   | Web server sending updates *to* a web browser client |
| **SSE Client** | `transport-sse`             | Server-Sent Events (Web)   | Web client connecting *to* an MCP web server         |
| **Child Process** | `transport-child-process`   | Standard Input/Output      | Client program launching and talking to a server program on the same machine |
| **TCP Stream** | (Works automatically)       | Network Sockets (TCP/IP)   | Direct network connection between computers          |
| **In-Process** (Maybe) | (Might be separate crate) | Inside the same program    | Testing, components talking within one application   |

* **SSE (Server-Sent Events):** A way for a web server to continuously send updates to a web browser over a regular HTTP connection. Good for web-based MCP tools.
* **Child Process:** When one program starts another program and communicates with it through its standard input and output pipes.

**C. Example: Setting up Stdio (Keyboard/Screen)**

This is often the simplest. You just use the standard input (keyboard) and standard output (screen) of your program.

```rust
// Make sure you 'use' these at the top of your file
use tokio::io::{stdin, stdout};

// stdin() gives you access to keyboard input (implements AsyncRead)
// stdout() gives you access to screen output (implements AsyncWrite)
// Putting them in a tuple (pair) like this automatically works as an IntoTransport!
let stdio_transport = (stdin(), stdout());

// Now you can give this 'stdio_transport' to the rmcp function that starts the server or client.
```
* **`use tokio::io::{stdin, stdout};`**: This line imports the `stdin` and `stdout` functions from the `tokio` library's input/output module. `use` statements bring code from other modules or crates into your current scope so you can use it.
* **`let stdio_transport = (stdin(), stdout());`**: This creates a variable named `stdio_transport`. It holds a **tuple** (a fixed-size collection of different types, written with parentheses `()`). In this case, the tuple contains the result of calling `stdin()` and `stdout()`. Because `stdin()` provides async reading and `stdout()` provides async writing, `rmcp` knows how to use this tuple as a communication channel thanks to the `IntoTransport` trait.

**D. Example: Setting up Child Process (Client Launches Server)**

This is useful when your client program needs to start the server program itself.

```rust
// Import necessary items
use rmcp::transport::TokioChildProcess;
use tokio::process::Command;
use std::error::Error; // For basic error handling

// async fn defines a function that can run asynchronously (can pause/resume)
async fn setup_child_process_transport() -> Result<TokioChildProcess, Box<dyn Error>> {
    // Command::new creates a command to run another program
    let command = Command::new("npx") // Example: using npx (comes with Node.js)
        .arg("-y") // Arguments passed to the command
        .arg("@modelcontextprotocol/server-everything"); // The specific server program to run

    // Create the special transport that handles starting the process
    // The '?' operator is a shortcut in Rust: if TokioChildProcess::new fails,
    // it immediately returns the error from this function.
    let child_process_transport = TokioChildProcess::new(command)?;

    // If successful, return the transport wrapped in Ok()
    // Result<OkType, ErrType> is Rust's way of handling operations that might fail.
    // Box<dyn Error> is a common way to handle various kinds of errors.
    Ok(child_process_transport)
}
```
* **`async fn ... -> Result<..., Box<dyn Error>>`**: Defines an asynchronous function that returns a `Result`. `Ok` contains a `TokioChildProcess` on success, `Err` contains a general error (`Box<dyn Error>`) on failure.
* **`Command::new("...")`**: Sets up the command to execute (here, `npx`).
* **`.arg("...")`**: Adds arguments to the command line.
* **`TokioChildProcess::new(command)?`**: Tries to create the transport by setting up the command. The `?` handles potential errors during setup (like the command not being found).
* **`Ok(...)`**: Wraps the successful result.

**E. Example: Setting up SSE (Web Communication)**

SSE is for web stuff.

**Client Connecting to SSE Server:**

```rust
use rmcp::transport::SseTransport;
use std::error::Error;

async fn setup_sse_client_transport() -> Result<SseTransport, Box<dyn Error>> {
    // The web address (URL) of the server's special SSE endpoint
    let server_url = "http://localhost:8000/sse"; // Just an example address

    // Tell the SseTransport to connect to that URL.
    // .await pauses the function here until the connection attempt finishes.
    let sse_transport = SseTransport::start(server_url).await?;
    Ok(sse_transport)
}
```
* **`let server_url = "..."`**: Stores the server's web address as a string.
* **`SseTransport::start(server_url).await?`**: This is the core part.
    * `SseTransport::start(...)`: Calls a function associated with the `SseTransport` type to begin the connection process. This likely returns something called a `Future` (a value that will be ready later).
    * `.await`: Pauses the `async fn` until the `Future` completes (i.e., until the connection is established or fails).
    * `?`: Handles any error that occurred during the connection attempt.

**Server Setup for SSE:** This usually involves using a Rust web framework (like Axum, Actix, Warp, or Poem). The web framework handles the incoming web connection, and when it sees a request for the `/sse` path, it hands the connection over to `rmcp`'s SSE transport handler. The `rmcp` examples likely show how to do this with a specific framework like Poem.

**V. Building an MCP Server with `rmcp`**

Let's build the "provider" part – the server that offers tools or data.

**A. Defining the Server's Memory (State)**

Your server might need to remember things between requests, like a user's score, database connection info, or settings. You define this "memory" in a **`struct`**.

```rust
use std::sync::{Arc, Mutex}; // Tools for safe sharing of data in async code

// Example: A server that just remembers a single number (a counter)
#[derive(Clone)] // Allows making copies of this state easily, often needed
struct CounterServerState {
    // Arc<Mutex<...>> is a way to safely share data that might be changed
    // by different tasks running at the same time.
    // Think of Mutex as a "talking stick" - only the task holding it can change the value.
    // Arc lets multiple tasks know where the data is.
    value: Arc<Mutex<i32>>,
    // You could add database connections, config settings, etc. here
}

impl CounterServerState { // 'impl' block defines methods (functions) for the struct
    // A function to create a new instance of our state
    fn new() -> Self { // 'Self' is a shorthand for 'CounterServerState' here
        Self {
            value: Arc::new(Mutex::new(0)), // Start the counter at 0
        }
    }
}

// --- OR ---

// Example: A simple calculator that doesn't need to remember anything between calls
#[derive(Debug, Clone)] // Debug allows printing, Clone allows copying
pub struct Calculator; // An empty struct, no fields needed!
```
* **`struct CounterServerState { ... }`**: Defines a custom data structure named `CounterServerState`.
* **`value: Arc<Mutex<i32>>`**: Defines a field named `value`. `i32` is a 32-bit integer. `Arc<Mutex<...>>` is an advanced Rust pattern for sharing mutable data safely across asynchronous tasks. For now, think of it as a safe box for our counter number.
* **`#[derive(Clone)]`**: An attribute that automatically adds the ability to make copies (`.clone()`) of `CounterServerState`.
* **`impl CounterServerState { ... }`**: Starts a block where we define functions (methods) associated with `CounterServerState`.
* **`fn new() -> Self { ... }`**: Defines a function `new` that takes no arguments (`()`) and returns a new instance of `CounterServerState` (`Self`). This is a common pattern for constructors.
* **`Arc::new(Mutex::new(0))`**: Creates the shared, mutable counter, initializing the inner value to `0`.
* **`struct Calculator;`**: Defines a struct with no fields. Useful when the server's actions don't depend on stored data. `pub` means it's public and can be used by other code.

**B. Implementing the Server's Brain (`ServerHandler`)**

Now we need to teach our `struct` how to act like an MCP server by implementing the `ServerHandler` trait (giving it the required skills).

```rust
use rmcp::{ServerHandler, model::*, service::RequestContext, error::Error as McpError};
use async_trait::async_trait; // Needed for using 'async fn' in traits

// Let's implement ServerHandler for our simple Calculator struct
#[async_trait] // Add this attribute when using async fn in the trait implementation
impl ServerHandler for Calculator { // We're giving 'Calculator' the 'ServerHandler' skills

    // Skill 1: Provide basic server info
    fn get_info(&self) -> ServerInfo { // '&self' means the method can access the struct's data (if any)
        ServerInfo {
            name: "CalculatorServer".into(), // .into() converts string literal to String
            version: "1.0.0".into(),
            instructions: Some("A simple calculator.".into()), // Optional instructions for the AI
            capabilities: None, // Optional: describe special server features
        }
    }

    // Skill 2: Handle requests from the client (The Core Logic!)
    // This function MUST be async fn because network/tool operations can take time
    async fn handle_request(
        &self,
        request: ClientRequest, // The message received from the client
        context: RequestContext<rmcp::service::RoleServer>, // Info about the connection
    ) -> Result<ServerResult, McpError> { // Must return Ok(ServerResult) or Err(McpError)

        // Use 'match' to handle different types of requests
        match request {
            ClientRequest::InitializeRequest(req) => {
                println!("Client connected: {:?}", req.params.client_info);
                // Send back confirmation and server info
                Ok(ServerResult::InitializeResult(InitializeResult {
                    server_info: self.get_info().into(), // Reuse our get_info method
                    capabilities: ServerCapabilities { /* ... specify if needed ... */ },
                }))
            }
            ClientRequest::ListToolsRequest(_) => {
                // This is often handled AUTOMATICALLY by the tool_box! macro later!
                // If handling manually, you'd build a list of Tool structs here.
                println!("Client asked for tools list");
                // Placeholder if not using tool_box! macro:
                Err(McpError::method_not_found("ListToolsRequest"))
            }
            ClientRequest::CallToolRequest(req) => {
                // This is also handled AUTOMATICALLY by the tool_box! macro!
                // If handling manually, you'd check req.params.name and call the right function.
                println!("Client wants to call tool: {}", req.params.name);
                // Placeholder if not using tool_box! macro:
                Err(McpError::method_not_found("CallToolRequest"))
            }
            // ... handle other requests like ListResourcesRequest, ReadResourceRequest etc.
            _ => { // Catch-all for requests we don't handle specifically
                 println!("Received an unknown request type");
                 Err(McpError::method_not_found("Unknown Request Type"))
            }
        }
    }

    // Skill 3: Handle notifications from the client (messages that don't need a reply)
    async fn handle_notification(
        &self,
        notification: ClientNotification,
    ) -> Result<(), McpError> { // Returns Ok(()) on success, or an error
        match notification {
            ClientNotification::CancelledNotification(note) => {
                println!("Client sent cancel for request ID: {:?}", note.params.request_id);
                // Add logic here to stop any long-running task if possible
                Ok(()) // Indicate success
            }
            // ... handle other notifications like ResourceSubscribed ...
            _ => {
                println!("Received an unhandled notification");
                Ok(())
            }
        }
    }
}
```
* **`#[async_trait]`**: A helper attribute needed when you use `async fn` inside a trait implementation.
* **`impl ServerHandler for Calculator`**: This line says: "We are now providing the implementation of the `ServerHandler` trait for the `Calculator` struct."
* **`fn get_info(&self) -> ServerInfo`**: Defines the `get_info` method required by the trait. `&self` gives read-only access to the `Calculator` instance. It returns a `ServerInfo` struct.
* **`async fn handle_request(...) -> Result<ServerResult, McpError>`**: Defines the asynchronous `handle_request` method. It takes the incoming `ClientRequest` and returns a `Result` which is either `Ok` containing a `ServerResult` (the response message) or `Err` containing an `McpError`.
* **`match request { ... }`**: A powerful Rust control flow construct. It checks what *variant* the `request` (which is an `enum` like `ClientRequest`) is and runs the code for that specific variant.
    * `ClientRequest::InitializeRequest(req) => { ... }`: If the request is `InitializeRequest`, the inner data (`req`) is extracted, and the code block runs.
    * `_ => { ... }`: The underscore `_` is a wildcard pattern, matching any variant not explicitly listed above.
* **`Ok(...)` / `Err(...)`**: Used to create the `Result` value that the function returns.
* **`async fn handle_notification(...) -> Result<(), McpError>`**: Defines the async notification handler. It returns `Ok(())` (an empty success value) or an `Err`. `()` is the "unit type" in Rust, representing no value.

**C. Defining Tools with Magic Spells (Macros)**

Manually handling `ListToolsRequest` and `CallToolRequest` in the `match` statement can be tedious. `rmcp` provides **macros** to make this super easy!

* **`#[tool]` Macro:** You put this **attribute** above a function (method) inside your `impl` block to mark it as an MCP tool that clients can call.
    * `description = "..."`: **Crucial!** This text describes what the tool does. The AI (LLM) uses this description to figure out *when* to use your tool.
    * `#[tool(param)]` or `#[tool(aggr)]`: Tells the macro how the function's arguments relate to the tool's input parameters. `aggr` is often used when parameters are grouped in a `struct`.
    * **Input Schema:** You usually define a `struct` for the tool's input parameters. Using `#[derive(Deserialize, JsonSchema)]` on that struct helps `rmcp` automatically figure out the expected input format (`input_schema`).
    * **Return Type:** The function should return something that `rmcp` can turn into a tool result. Often, this is `Result<SuccessType, ErrorType>`, where both types can be converted into `Contents` (like text).

* **`#[tool_box]` Macro:** This macro does two things:
    1.  Put it **above the `impl` block** that contains your `#[tool]` methods. It tells `rmcp` this block holds a collection of tools.
    2.  Use `tool_box!(@impl ...)` **inside your `ServerHandler` implementation**. This magic macro automatically writes the `match` arms for `ListToolsRequest` and `CallToolRequest` for you! It gathers info from your `#[tool]` methods and routes calls correctly.

**Example using Macros:**

```rust
use rmcp::{tool, tool_box, ServerHandler, model::*, error::Error as McpError, schemars};
use serde::{Deserialize, Serialize}; // For parameter structs
use schemars::JsonSchema; // To describe parameter structs
use std::fmt::{self, Display}; // For creating simple text results/errors
use async_trait::async_trait;

// --- Define Input/Output Structures ---

// 1. Define the input parameters for the 'sum' tool as a struct
#[derive(Deserialize, JsonSchema, Debug)] // Derive traits needed by rmcp/serde
pub struct SumRequest {
    #[schemars(description = "The first number to add")] // Good practice: describe fields
    pub a: i32,
    #[schemars(description = "The second number to add")]
    pub b: i32,
}

// 2. Define a simple way to return text results (could be more complex JSON too)
// We need to tell rmcp how to turn our result (like a String) into MCP's 'Contents' type.
// rmcp might handle String automatically, but let's be explicit for learning.
impl IntoContents for String {
    fn into_contents(self) -> Contents {
        Contents::Text(self) // Wrap the String in Contents::Text
    }
}

// 3. Define a simple error type we can return from tools
#[derive(Debug)] // Allows printing the error
struct CalculationError(String); // Wrap the error message in a struct

// Also tell rmcp how to turn our error into 'Contents'
impl IntoContents for CalculationError {
    fn into_contents(self) -> Contents {
        Contents::Error(ErrorContents { // Use Contents::Error for tool errors
            code: "CALC_ERROR".to_string(), // Give it an error code
            message: self.0, // Use the wrapped string as the message
            data: None, // Optional extra data
        })
    }
}

// --- Define the Server State/Logic ---

#[derive(Debug, Clone)]
pub struct Calculator; // Our stateless calculator again

// Use #[tool_box] on the impl block containing the tools
#[tool_box]
impl Calculator {
    // Mark this function as a tool with #[tool]
    #[tool(description = "Calculate the sum of two integers. Handles overflow.")]
    // 'async fn' because calculations *could* be complex (though not here)
    // Use #[tool(aggr)] because parameters are aggregated in SumRequest struct
    async fn sum(&self, #[tool(aggr)] req: SumRequest) -> Result<String, CalculationError> {
        // The actual logic
        match req.a.checked_add(req.b) { // checked_add prevents crashes on overflow
            Some(result) => Ok(result.to_string()), // Success: return the sum as a String
            None => Err(CalculationError("Addition overflowed!".to_string())), // Error
        }
        // rmcp + IntoContents handle turning Ok(String) or Err(CalcError) into CallToolResult
    }

    #[tool(description = "Calculate the difference of two integers (a - b). Handles overflow.")]
    // This tool takes parameters directly using #[tool(param)]
    // We also add #[derive(JsonSchema)] implicitly via schemars below
    fn subtract(
        &self,
        #[tool(param)] #[schemars(description = "The number to subtract from")] a: i32,
        #[tool(param)] #[schemars(description = "The number to subtract")] b: i32,
    ) -> Result<String, CalculationError> { // Returns Result<String, CalcError>
        match a.checked_sub(b) {
            Some(result) => Ok(result.to_string()),
            None => Err(CalculationError("Subtraction overflowed!".to_string())),
        }
    }
}

// --- Implement ServerHandler Using the Macros ---

#[async_trait]
#[tool_box] // Also need #[tool_box] here to link the handler to the tools impl block
impl ServerHandler for Calculator {
    fn get_info(&self) -> ServerInfo { /* ... same as before ... */
        ServerInfo {
             name: "CalculatorServer".into(),
             version: "0.1.0".into(),
             instructions: Some("A calculator with 'sum' and 'subtract' tools.".into()),
             ..Default::default() // Fills remaining fields with default values
         }
    }

    // This MAGIC line uses the macro to handle tool requests automatically!
    // It finds the 'sum' and 'subtract' methods in the #[tool_box] impl Calculator block.
    tool_box!(@impl ServerHandler for Calculator { sum, subtract });

    // We can still manually handle OTHER requests/notifications if needed:
    // async fn handle_request(&self, request: ClientRequest, ...) {
    //     match request {
    //         // Handle Initialize, Ping, etc. manually if the macro doesn't.
    //         // The macro adds arms for ListTools and CallTool implicitly here.
    //         _ => { /* ... */ }
    //     }
    // }
    // async fn handle_notification(...) { /* ... */ }
}

```
* **`#[derive(Deserialize, JsonSchema, Debug)]`**: Attributes applied to the `SumRequest` struct.
    * `Deserialize`: Allows `serde` to create this struct from incoming JSON data.
    * `JsonSchema`: Allows `schemars` (and thus `rmcp`) to automatically generate a description (schema) of what this struct looks like, which is needed for the tool definition.
    * `Debug`: Allows printing the struct for debugging using `{:?}`.
* **`#[schemars(description = "...")]`**: Provides descriptions for struct fields, which helps the AI understand the tool parameters.
* **`impl IntoContents for String { ... }`**: Implements the `IntoContents` trait for Rust's `String` type. This tells `rmcp` how to convert a `String` into the standard `Contents::Text` variant used in MCP messages. `rmcp` might provide this automatically, but showing it helps understand the concept.
* **`struct CalculationError(String);`**: Defines a simple custom error type containing just a message `String`.
* **`impl IntoContents for CalculationError { ... }`**: Teaches `rmcp` how to turn our custom error into `Contents::Error`, including a code and message. This allows tools to return structured errors.
* **`#[tool_box]`**: Placed *before* `impl Calculator { ... }` to mark it as the container for our tool methods (`sum`, `subtract`).
* **`#[tool(description = "...")]`**: Placed *before* each tool method (`sum`, `subtract`) to define it as an MCP tool and provide the crucial description.
* **`#[tool(aggr)] req: SumRequest`**: In `sum`, this indicates that all parameters for the tool are aggregated (grouped) within the `SumRequest` struct named `req`. `rmcp` will expect the client to send parameters matching this struct.
* **`#[tool(param)] #[schemars(...)] a: i32`**: In `subtract`, `#[tool(param)]` indicates that `a` is a direct, individual parameter of the tool. `#[schemars(...)]` provides its description.
* **`checked_add`, `checked_sub`**: Safe arithmetic methods in Rust that return `Some(result)` on success and `None` if an overflow (number too big/small) would occur. This prevents crashes.
* **`Result<String, CalculationError>`**: The return type for both tools. Indicates success (`Ok`) with a `String` or failure (`Err`) with our `CalculationError`. Because both `String` and `CalculationError` implement `IntoContents`, `rmcp` knows how to package this into the final `CallToolResult` message.
* **`#[tool_box]` before `impl ServerHandler for Calculator`**: Links this handler implementation to the tool definitions found in the *other* `#[tool_box]` block.
* **`tool_box!(@impl ServerHandler for Calculator { sum, subtract });`**: The core macro magic! This line expands into the necessary `match` arms within `handle_request` to:
    * Respond to `ListToolsRequest` by creating a list based on the `#[tool]` descriptions of `sum` and `subtract`.
    * Respond to `CallToolRequest` by checking the requested tool name and calling the correct method (`sum` or `subtract`) with the provided parameters.

**D. Offering Data (Resources) - The Idea**

MCP also lets servers offer **Resources**, which are basically pieces of data that clients can read or even subscribe to (get updates when the data changes). Think of files, database entries, or sensor readings.

* **How it might work in `rmcp` (Conceptually):**
    * You'd likely need to manually implement the `handle_request` logic for `ClientRequest` types like `ListResourcesRequest`, `ReadResourceRequest`, `SubscribeResourceRequest`, etc.
    * Your server `struct` would need to manage access to this data (e.g., read files, query databases).
    * To send updates for subscribed resources, you'd use the connection object (`Peer`) provided in the `RequestContext` to send `notifications/resources/updated` messages back to the client.
* **Finding Examples:** The provided text notes that clear `rmcp` examples for resources were scarce in its sources. You'd need to check the **official `rmcp` examples folder** in its GitHub repository or the API documentation on docs.rs for concrete code.

**E. Starting the Server**

Once you have your server state (`struct`), the handler logic (`impl ServerHandler`), and chosen a transport, you can start the server running.

```rust
use rmcp::ServiceExt; // Provides the .serve() method
use tokio::io::{stdin, stdout}; // For stdio transport

// Assume 'Calculator' struct and its 'ServerHandler' impl are defined above

#[tokio::main] // Marks the main function as the entry point for the Tokio async runtime
async fn main() -> Result<(), Box<dyn std::error::Error>> { // Standard async main signature
    // 1. Create an instance of your server handler/state
    let calculator_service = Calculator;

    // 2. Create the transport you want to use
    let transport = (stdin(), stdout()); // Using stdio for this example

    println!("Starting Calculator MCP Server on stdio...");

    // 3. Start the server!
    // .serve() takes the handler and the transport, starts listening,
    // handles the initial MCP handshake, and runs the message loop.
    // .await pauses until the initial connection/handshake is done (or fails).
    let server_peer = calculator_service.serve(transport).await?;
    // 'server_peer' represents the server's view of the connection to the client.

    println!("Server connected to a client and running. Waiting for shutdown...");

    // 4. Keep the server running
    // .waiting().await pauses the main function until the connection is closed
    // (e.g., client disconnects, error occurs, or shutdown signal).
    let shutdown_reason = server_peer.waiting().await?;
    println!("Server shut down. Reason: {:?}", shutdown_reason);

    Ok(()) // Indicate main function finished successfully
}
```
* **`#[tokio::main]`**: An attribute macro that sets up the asynchronous runtime and makes your `async fn main` the starting point.
* **`async fn main() -> Result<(), Box<dyn std::error::Error>>`**: The standard way to write a `main` function in Rust that uses async operations and might return errors. `Ok(())` means success.
* **`let calculator_service = Calculator;`**: Creates an instance of our `Calculator` struct (which implements `ServerHandler`).
* **`let transport = (stdin(), stdout());`**: Creates the Stdio transport.
* **`calculator_service.serve(transport).await?`**: This is the key line to start the server.
    * `.serve()`: A method (likely from the `ServiceExt` trait) called on our handler. It takes ownership of the handler and the transport. It starts the MCP communication loop (listening for messages, calling your handler methods).
    * `.await`: Since `serve` is async (it involves waiting for connections/messages), we `.await` its completion (which usually means the initial handshake is done).
    * `?`: Handles any error during startup (e.g., transport issue, handshake failure).
    * `let server_peer = ...`: The `serve` method returns a `Peer` object, which represents the active connection from the server's perspective.
* **`server_peer.waiting().await?`**: This keeps the `main` function alive while the server is running. It waits asynchronously until the `Peer` (connection) is closed for any reason. It returns the reason for the shutdown.

**VI. Building an MCP Client with `rmcp`**

Now let's build the "requester" – the client that connects to a server to use its tools or resources.

**A. Defining the Client's Brain (Handler)**

Like the server, the client *can* have state and needs a `struct` that implements the `ClientHandler` trait. But often, clients are simpler.

* **Super Simple Handler:** If your client mainly just sends requests and doesn't need to react to server notifications (like progress updates), you can use the empty tuple `()` as your handler!
    ```rust
    let handler = (); // The simplest possible handler
    ```
* **Slightly Simple Handler:** If you just need to provide basic client info, you can sometimes use the `ClientInfo` struct itself as the handler.
* **Custom Handler:** If your client *does* need to react to server notifications (like `on_progress` or `on_resource_updated`) or handle server-initiated requests, you'll need to define your own `struct` and implement `ClientHandler` for it, similar to how we did for the server, but implementing the client-side methods.

**B. Connecting to the Server**

The client uses a transport to connect to the server's address and starts the process using `.serve()`.

```rust
use rmcp::{ServiceExt, transport::SseTransport, ClientHandler, ClientInfo, model::Peer, service::RoleClient};
use std::error::Error;

// Example using SSE transport and the simple () handler

#[tokio::main]
async fn main() -> Result<(), Box<dyn Error>> {
    // 1. Server's address (using SSE in this example)
    let server_url = "http://localhost:8000/sse";

    // 2. Create the transport
    println!("Creating SSE transport to {}...", server_url);
    let transport = SseTransport::start(server_url).await?;

    // 3. Create the handler (using the simplest one here)
    let handler = ();

    println!("Connecting to server...");

    // 4. Connect!
    // Call .serve() on the handler with the transport.
    // This does the MCP handshake from the client side.
    // It returns a 'Peer' object representing the client's connection.
    // We specify RoleClient to get the client-side Peer type.
    let client_peer: Peer<RoleClient> = handler.serve(transport).await?;

    println!("Successfully connected to server!");

    // Now you can use 'client_peer' to talk to the server...
    // (Code from next section goes here)

    Ok(())
}
```
* **`let client_peer: Peer<RoleClient> = handler.serve(transport).await?;`**: This is the client-side connection call.
    * `handler.serve(transport)`: Similar to the server, but called on the client handler. It initiates the MCP handshake with the server over the given transport.
    * `.await?`: Waits for the handshake to complete successfully.
    * `let client_peer: Peer<RoleClient>`: Stores the resulting connection object. `Peer<RoleClient>` is the type representing the client's view of the connection. This `client_peer` is your remote control for talking to the server.

**C. Talking to the Server (Using the `Peer`)**

Once connected, you use the `client_peer` object to interact with the server. All these methods are `async` and return a `Result`.

* **Listing things:**
    * `client_peer.list_tools(ListToolsParams {}).await?`: Asks the server for its list of available tools. (Might take `None` or an empty struct depending on `rmcp` version).
    * `client_peer.list_resources(...).await?`: Asks for available data resources.
    * `client_peer.list_roots(...).await?`: Might ask for top-level resources.

* **Calling tools:**
    * `client_peer.call_tool("tool_name", parameters).await?`: Tells the server to run a specific tool.
        * `"tool_name"`: The name of the tool you want to run (must match the name on the server).
        * `parameters`: The input data for the tool. This needs to be in a format `serde_json` can handle, often created using the `json!` macro: `let params = json!({ "a": 10, "b": 5 });`. The structure must match what the server's tool expects (like our `SumRequest` struct earlier).

* **Reading data:**
    * `client_peer.read_resource("resource_uri").await?`: Asks the server for the content of a specific resource (e.g., `read_resource("file:///data.txt").await?`).

* **Sending simple messages (Notifications):**
    * `client_peer.notify_cancelled(...).await?`: Tells the server to cancel a previous request if possible.

* **Handling Responses:** Each `.await?` call will either give you `Ok(ResponseStruct)` (like `Ok(ListToolsResult)` or `Ok(CallToolResult)`) containing the server's answer, or `Err(McpError)` if something went wrong. You need to check the result using `match` or `.unwrap()/.expect()` (carefully!).

**D. Handling Server Messages (Notifications/Requests)**

If the server might send *unsolicited* messages (like progress updates or resource changes), your client needs a proper `ClientHandler` implementation (not just `()`). When the server sends such a message, `rmcp` will automatically call the corresponding method on your handler (e.g., `on_progress`, `on_resource_updated`, `handle_request`). You write the code inside those methods to react appropriately.

**E. Example: Simple Client (Launching Server)**

This client connects to a server it launches itself (using Child Process transport), lists the tools, and calls one.

```rust
use rmcp::{
    service::{Peer, RoleClient}, // Core service types
    transport::TokioChildProcess, // The child process transport
    ClientHandler, ClientInfo, ServiceExt, // Client handler trait, info struct, .serve() method
    model::* // Import all model types (requests, results etc.) '*' is a wildcard
};
use tokio::process::Command; // To create the server command
use serde_json::json; // For easily creating JSON parameters
use std::error::Error;
use async_trait::async_trait;

// --- Define a Client Handler (Slightly more than () ) ---
// This handler stores the client's info and the connection Peer
#[derive(Clone, Debug)] // Cloneable and printable
struct MyClientHandler {
    info: ClientInfo,
    // Option means it might or might not have a Peer yet
    peer: Option<Peer<RoleClient>>,
}

impl MyClientHandler {
    fn new() -> Self {
        Self {
            info: ClientInfo {
                name: "my-cool-client".into(),
                version: "0.1.0".into(),
                supported_protocol_versions: None, // Let rmcp handle default versions
            },
            peer: None, // Starts without a connection
        }
    }
}

#[async_trait]
impl ClientHandler for MyClientHandler {
    // Skill 1: Provide client info
    fn get_info(&self) -> ClientInfo { self.info.clone() } // Return a copy of the info

    // Skills to manage the connection Peer
    fn get_peer(&self) -> Option<Peer<RoleClient>> { self.peer.clone() }
    fn set_peer(&mut self, peer: Peer<RoleClient>) { // '&mut self' needed to modify the struct
        self.peer = Some(peer); // Store the connection peer when established
    }

    // --- Optional: Handle Notifications from Server ---
    async fn on_progress( // Example: React to progress updates
        &self,
        progress: ProgressParams,
    ) -> Result<(), McpError> {
        println!(
            "Server Progress: Request({:?}), Token({:?}), Message: {:?}",
            progress.request_id, progress.progress_token, progress.message
        );
        Ok(())
    }

     async fn on_resource_updated( // Example: React to data changes
         &self,
         params: ResourceUpdatedParams,
     ) -> Result<(), McpError> {
         println!("Server notified: Resource '{}' updated!", params.uri);
         // Maybe trigger a re-read of the resource here?
         Ok(())
     }

    // Implement handle_request if you expect the server to send requests TO the client
}


#[tokio::main]
async fn main() -> Result<(), Box<dyn Error>> {
    // 1. Build Transport (Child Process)
    println!("Spawning server process...");
    let transport = TokioChildProcess::new(
        Command::new("npx") // Command to run the server (adjust path if needed)
            .arg("-y")
            .arg("@modelcontextprotocol/server-everything") // The server package
            // Ensure we capture the server's output/input streams
            .stdout(std::process::Stdio::piped())
            .stderr(std::process::Stdio::piped()) // Good to capture errors too
            .stdin(std::process::Stdio::piped())
    )?;

    // 2. Build Handler
    let handler = MyClientHandler::new();

    // 3. Connect (Serve the client handler over the transport)
    println!("Connecting to server via child process transport...");
    // .serve() will call handler.set_peer() when connected
    let client_peer = handler.serve(transport).await?;
    println!("Client connected successfully!");

    // 4. Interact with Server using the 'client_peer' handle

    // --- List Tools ---
    println!("Listing available tools...");
    match client_peer.list_tools(ListToolsParams {}).await { // Call list_tools
        Ok(tools_result) => { // If successful (Ok)
            println!("Available tools:");
            for tool in tools_result.tools { // Loop through the list of tools
                println!("  - Name: {}", tool.name);
                // Use .unwrap_or_default() for optional fields like description
                println!("    Desc: {}", tool.description.unwrap_or_default());
            }
        }
        Err(e) => { // If there was an error (Err)
            eprintln!("Error listing tools: {}", e); // Print the error
        }
    }

    // --- Call a Tool (e.g., 'sum' from our server example) ---
    println!("Calling tool 'sum' with a=10, b=5...");
    // Create parameters using json! macro
    let params = json!({ "a": 10, "b": 5 });
    match client_peer.call_tool("sum", params).await { // Call the tool by name
        Ok(result) => {
            println!("Tool 'sum' result: {:?}", result.contents); // Print the result content
        }
        Err(e) => {
            eprintln!("Error calling tool 'sum': {}", e);
        }
    }

    // --- Example: Try calling 'subtract' tool ---
     println!("Calling tool 'subtract' with a=100, b=33...");
     let params = json!({ "a": 100, "b": 33 }); // Parameters match the 'subtract' function args
     match client_peer.call_tool("subtract", params).await {
         Ok(result) => println!("Tool 'subtract' result: {:?}", result.contents),
         Err(e) => eprintln!("Error calling tool 'subtract': {}", e),
     }

    // --- Read a Resource (Hypothetical Example) ---
    // println!("Reading resource 'file:///example.txt'...");
    // match client_peer.read_resource("file:///example.txt").await {
    //     Ok(resource_result) => println!("Resource content: {:?}", resource_result.contents),
    //     Err(e) => eprintln!("Error reading resource: {}", e),
    // }

    // 5. Wait for Shutdown (Keep client running to receive notifications)
    println!("Client running. Waiting for shutdown signal or server disconnect...");
    let reason = client_peer.waiting().await?; // Wait for the connection to end
    println!("Client shutdown: {:?}", reason);

    Ok(())
}
```
* **`MyClientHandler` struct**: Holds `ClientInfo` and an `Option<Peer<RoleClient>>`. The `Peer` is `None` initially and gets filled in by `rmcp` via the `set_peer` method when the connection succeeds.
* **`impl ClientHandler for MyClientHandler`**: Implements the required methods (`get_info`, `get_peer`, `set_peer`) and optionally `on_progress`, `on_resource_updated`, etc., to react to server messages.
* **`Command::new(...).stdout(...).stderr(...).stdin(...)`**: Configures the command to ensure the client can communicate with the child process's standard input, output, and error streams. `piped()` is crucial.
* **`handler.serve(transport).await?`**: Connects, performing the handshake. The `Peer` object returned is stored in `client_peer`. `rmcp` internally calls `handler.set_peer()` upon successful connection.
* **`client_peer.list_tools(...).await`**: Calls the server's `list_tools` method asynchronously.
* **`match ... { Ok(...) => {...}, Err(...) => {...} }`**: The standard Rust way to handle the `Result` returned by async calls. Check if it succeeded (`Ok`) or failed (`Err`) and act accordingly.
* **`tools_result.tools`**: The `Ok` variant contains a result struct (e.g., `ListToolsResult`) which has a field (`tools`) containing the actual data (a list of `Tool` structs).
* **`json!({ "key": value, ... })`**: A convenient macro from `serde_json` to create `serde_json::Value` objects (which represent arbitrary JSON data) easily. This is used to build the parameters for `call_tool`.
* **`client_peer.call_tool("sum", params).await`**: Calls the specific tool named "sum" on the server, sending the `params`.
* **`result.contents`**: The `Ok` variant from `call_tool` contains a `CallToolResult` struct, which has a `contents` field holding the actual result data sent back by the tool (as `Contents`).
* **`client_peer.waiting().await?`**: Keeps the client running until the connection ends, allowing time for server notifications to be received and handled by the `ClientHandler` methods (`on_progress`, etc.).

**VII. Handling Mistakes (Error Handling) in `rmcp`**

Computers aren't perfect, and networks can be unreliable. Good programs need to handle errors gracefully.

**A. The Main Error Type: `McpError`**

When something goes wrong within the `rmcp` toolkit or the MCP communication, it usually signals this using its main error type: `rmcp::error::Error` (often shortened to `McpError`). This error type can represent different kinds of problems:

* **Protocol Errors:** Like getting a message that doesn't follow the MCP rules, or messages arriving in the wrong order.
* **Transport Errors:** Problems with the communication channel itself – the network connection dropped, couldn't connect, data got garbled.
* **Application Errors:** Errors that happen inside *your* server code (like a tool failing) that get reported back through MCP.
* **Initialization Errors:** The initial client-server handshake failed.

**B. Handling Errors in Your Handlers (`ServerHandler`/`ClientHandler`)**

* **In `handle_request`:** Your code inside `handle_request` (on both client and server) might fail (e.g., database error, file not found). You need to catch these errors and return an `Err(McpError)` or construct a proper MCP error response to send back.
* **In `handle_notification`:** If your code processing a notification fails, you should return an `Err(McpError)`. This might signal a bigger problem to `rmcp`.
* **In Tools (`#[tool]` functions):** Returning `Result<Success, Error>` from your tool functions is the best way. If you return `Err(YourErrorType)`, and `YourErrorType` implements `IntoContents` (like our `CalculationError` example), `rmcp` automatically packages it as an error result for the client.

**C. Handling Errors When Using the `Peer`**

Every time you call a method on the `client_peer` (like `call_tool`, `list_tools`, `read_resource`), it returns a `Result<..., McpError>`. You *must* check this result!

```rust
// Using match (safer)
match client_peer.call_tool("sum", params).await {
    Ok(result) => { /* Handle successful result */ }
    Err(e) => { /* Handle the McpError 'e' */ }
}

// Using ? (convenient if the current function also returns Result<_, McpError>)
// This will automatically return the error 'e' from the current function if call_tool fails.
let result = client_peer.call_tool("sum", params).await?;
// If we reach here, it was Ok. Use 'result'.
```

**D. Being Robust (Building Stronger Apps)**

* **Timeouts:** Don't wait forever for a response. Add timeouts, especially for network calls.
* **Retries:** If a network error seems temporary, maybe try again after a short pause.
* **Logging:** Write messages about what your program is doing and any errors it encounters. This helps find bugs later (the `tracing` crate is popular in async Rust).
* **Clear Errors:** When your server tool fails, send back a meaningful error message to the client using the `Result<_, YourError>` pattern.

**VIII. The End? Nope, the Beginning!**

So, that's the whirlwind tour of the official `rmcp` toolkit for using the Model Context Protocol in Rust!

It gives you a solid, official way to let your Rust programs talk to smart AIs (LLMs) and the tools/data they need, using the power of asynchronous programming with Tokio.

**Recap of `rmcp`'s Superpowers:**

* **Official & Supported:** It's the main one!
* **Async Power:** Uses Tokio to handle lots of things at once.
* **Flexible Talking:** Supports different communication pipes (Stdio, SSE, etc.) via `IntoTransport`.
* **Easy Tools:** Macros (`#[tool]`, `#[tool_box]`) make adding server tools much simpler.
* **Customizable:** Built with traits, so you can extend it.

**Where to Go Next?**

* **Official MCP Website:** `modelcontextprotocol.io` (Learn the big ideas).
* **MCP Rules (Specification):** `spec.modelcontextprotocol.io` (The detailed rulebook).
* **`rmcp` Code & Examples:** `github.com/modelcontextprotocol/rust-sdk` (Look in the `examples/` folder!).
* **`rmcp` Detailed Docs:** `docs.rs/rmcp` (Reference for every function, struct, trait).
* **MCP Community:** Maybe a Discord server or GitHub discussions for asking questions.

This guide gave you the basics. Now you can explore the examples, try building a simple client or server, and start connecting your Rust code to the world of AI! Good luck, and have fun coding!