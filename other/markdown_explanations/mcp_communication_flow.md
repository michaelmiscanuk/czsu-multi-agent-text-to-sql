# MCP Communication Flow in Our Implementation

This section describes how the Model Context Protocol (MCP) is implemented in our text-to-SQL application. The following paragraphs walk through the communication flow step by step, from the moment a user asks a question until they receive their final answer. The description follows the architecture shown in the accompanying diagram (mcp_communication_flow_v6.drawio), which illustrates the four-layer structure: MCP Host (AI Application), MCP Client, MCP Server, and Database.

---

## User Prompt

Everything starts when the user types their question into the application. This could be something like "What is the unemployment rate in Prague?" or "Show me hotel statistics for 2023". The question enters our system through the frontend interface and is passed to the AI application for processing. At this point, the user's natural language question is just a string of text that needs to be transformed into actionable database queries.

---

## Generate Query Node

The user's question first arrives at the `generate_query_node()` function, which is a node in our LangGraph workflow. This node is responsible for setting up everything needed to answer the question. It loads the relevant database schemas so the LLM knows what tables and columns are available, builds a comprehensive system prompt with instructions on how to generate SQL queries, and initializes the language model. Most importantly, it binds the MCP tools to the LLM, giving the AI model the ability to execute database queries. Once everything is set up, the node kicks off an agentic loop where the LLM will autonomously decide what queries to run.

---

## Agentic Loop

The agentic loop is where the actual intelligence happens. The LLM examines the user's question along with the available database schemas and generates SQL queries to gather the necessary data. What makes this "agentic" is that the LLM doesn't just generate one query and stop – it can call the query tool multiple times, examining the results after each call and deciding whether it needs more information. If the first query doesn't return enough data, it generates another one. If it needs to check something in a different table, it does that too. This continues until the LLM decides it has gathered sufficient data to answer the user's question, at which point it signals completion and the loop ends.

---

## Get SQLite Tools

Before the agentic loop can execute any queries, it needs access to the database tools. The `get_sqlite_tools()` function handles this by reading the MCP server URL from the environment configuration and establishing the connection. This function also implements a fallback mechanism – if the remote MCP server is unavailable for any reason, it can fall back to using a local SQLite database instead. The function returns a list of tools that the LLM can use, with the primary tool being `sqlite_query` for executing SQL queries.

---

## MultiServerMCPClient

The actual connection to the MCP server is handled by the `MultiServerMCPClient` from the langchain-mcp-adapters library. This client establishes an HTTP connection to our FastMCP server using the streamable-http transport. Once connected, it retrieves the list of available tools from the server – in our case, just the `sqlite_query` tool. When the LLM in the agentic loop decides to execute a query, it calls `tool.ainvoke()` on this client, which packages the SQL query into an MCP protocol request and sends it over HTTP to the server.

---

## HTTP Request (MCP Protocol)

When the client invokes a tool, the request travels over the network using the MCP protocol encapsulated in HTTP. The request contains the SQL query that the LLM generated, formatted according to the MCP specification. This is where the standardization of MCP really pays off – the client and server communicate using a well-defined protocol, so they can be developed, deployed, and scaled independently. The request is sent to port 8100 where our FastMCP server is listening.

---

## FastMCP Server

On the receiving end, our FastMCP server (named "CZSU-SQLite-Server") accepts the incoming HTTP request. The server is built using the FastMCP library, which handles all the MCP protocol details automatically. When a request arrives, the server parses it to determine which tool is being called and with what parameters. The server exposes tools through the MCP protocol using the `@mcp.tool()` decorator, making them discoverable and callable by any MCP-compatible client. In our implementation, we expose a single tool for querying the database.

---

## Route to Tool

Once the server receives the request, it routes it to the appropriate tool handler. In our case, all query requests go to the `sqlite_query` tool, which is decorated with `@mcp.tool()`. This routing is handled automatically by FastMCP based on the tool name specified in the request. The tool receives the SQL query string along with a context object that allows it to log information about the execution.

---

## @mcp.tool() sqlite_query

The `sqlite_query` tool is the heart of our MCP server. It's an async function that receives the SQL query string and executes it against the database. The tool first logs the incoming query for debugging purposes, then establishes a connection to the database. The actual query execution happens in a separate thread (using `asyncio.to_thread()`) to avoid blocking the async server. After the query completes, the results are formatted as a JSON string and returned. This formatting ensures that the LLM can easily parse and understand the results.

---

## Execute Query

The tool handler connects to our Turso database, which is a cloud-hosted SQLite database. The connection is established using the libsql library with an authentication token for security. Once connected, the SQL query is executed using a standard cursor pattern – `cursor.execute()` runs the query, and `cursor.fetchall()` retrieves all the results. The database contains statistical data from the Czech Statistical Office, organized across many tables covering topics like employment, demographics, tourism, and more.

---

## Query Results (Return Path)

After the database executes the query, the results flow back through the same layers in reverse. The raw query results (rows and columns) are first formatted into a string representation. If there are no results, the tool returns "No results found". For a single value, it returns just that value as a string. For multiple rows, it formats the data as JSON to preserve the structure. This formatted result is then returned from the tool handler back to the FastMCP server.

---

## Tool Response and HTTP Response

The FastMCP server packages the tool's return value into an MCP protocol response and sends it back over HTTP to the waiting client. The response includes the query results as a JSON string, which preserves all the data the LLM needs to understand what the database returned. The HTTP response travels back over the network to the MCP client that initiated the request.

---

## Tool Result to Agentic Loop

The `MultiServerMCPClient` receives the HTTP response, extracts the tool result, and passes it back to the agentic loop. At this point, the LLM examines the results to decide what to do next. If the data is sufficient to answer the user's question, it signals completion. If not, it generates another SQL query and the whole process from step 5 onwards repeats. This iterative process continues until the LLM has gathered all the information it needs.

---

## Final Answer

Once the agentic loop completes, all the gathered query results are passed to an answer formatting node. This node combines the SQL results with any relevant PDF documentation chunks that were retrieved in parallel, and synthesizes a comprehensive answer to the user's original question. The formatted answer is then displayed to the user through the frontend interface, completing the full communication cycle.

---

## Summary

Our MCP implementation demonstrates how the protocol enables clean separation of concerns in an AI application. The AI agent (MCP Host) focuses on understanding questions and deciding what data to gather. The MCP Client handles protocol communication. The MCP Server manages tool execution and database access. And the database stores the actual data. Each layer can be developed, tested, and deployed independently, which is exactly the kind of modularity and interoperability that MCP was designed to provide.
