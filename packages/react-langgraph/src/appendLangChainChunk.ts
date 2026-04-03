import {
  InputTokenDetails,
  LangChainMessage,
  LangChainMessageChunk,
  MessageContentText,
  ModalitiesTokenDetails,
  OutputTokenDetails,
  UsageMetadata,
} from "./types";
import { parsePartialJsonObject } from "assistant-stream/utils";

function mergeModalitiesTokenDetails(
  a?: ModalitiesTokenDetails,
  b?: ModalitiesTokenDetails,
): ModalitiesTokenDetails {
  const output: ModalitiesTokenDetails = {};
  if (a?.audio !== undefined || b?.audio !== undefined) {
    output.audio = (a?.audio ?? 0) + (b?.audio ?? 0);
  }
  if (a?.image !== undefined || b?.image !== undefined) {
    output.image = (a?.image ?? 0) + (b?.image ?? 0);
  }
  if (a?.video !== undefined || b?.video !== undefined) {
    output.video = (a?.video ?? 0) + (b?.video ?? 0);
  }
  if (a?.document !== undefined || b?.document !== undefined) {
    output.document = (a?.document ?? 0) + (b?.document ?? 0);
  }
  if (a?.text !== undefined || b?.text !== undefined) {
    output.text = (a?.text ?? 0) + (b?.text ?? 0);
  }
  return output;
}

function mergeInputTokenDetails(
  a?: InputTokenDetails,
  b?: InputTokenDetails,
): InputTokenDetails {
  const output: InputTokenDetails = {
    ...mergeModalitiesTokenDetails(a, b),
  };
  if (a?.cache_read !== undefined || b?.cache_read !== undefined) {
    output.cache_read = (a?.cache_read ?? 0) + (b?.cache_read ?? 0);
  }
  if (a?.cache_creation !== undefined || b?.cache_creation !== undefined) {
    output.cache_creation = (a?.cache_creation ?? 0) + (b?.cache_creation ?? 0);
  }
  return output;
}

function mergeOutputTokenDetails(
  a?: OutputTokenDetails,
  b?: OutputTokenDetails,
): OutputTokenDetails {
  const output: OutputTokenDetails = {
    ...mergeModalitiesTokenDetails(a, b),
  };
  if (a?.reasoning !== undefined || b?.reasoning !== undefined) {
    output.reasoning = (a?.reasoning ?? 0) + (b?.reasoning ?? 0);
  }
  return output;
}

function mergeUsageMetadata(
  a?: UsageMetadata,
  b?: UsageMetadata,
): UsageMetadata {
  return {
    input_tokens: (a?.input_tokens ?? 0) + (b?.input_tokens ?? 0),
    output_tokens: (a?.output_tokens ?? 0) + (b?.output_tokens ?? 0),
    total_tokens: (a?.total_tokens ?? 0) + (b?.total_tokens ?? 0),
    input_token_details: mergeInputTokenDetails(
      a?.input_token_details,
      b?.input_token_details,
    ),
    output_token_details: mergeOutputTokenDetails(
      a?.output_token_details,
      b?.output_token_details,
    ),
  };
}

/**
 * Merges an AIMessageChunk into a previous message. Chunks must have
 * `type: "AIMessageChunk"` — JS LangGraph servers send `type: "ai"`,
 * so callers should normalize the type before passing chunks here.
 */
export const appendLangChainChunk = (
  prev: LangChainMessage | undefined,
  curr: LangChainMessage | LangChainMessageChunk,
): LangChainMessage => {
  if (curr.type !== "AIMessageChunk") {
    return curr;
  }

  if (!prev || prev.type !== "ai") {
    return {
      ...curr,
      type: curr.type.replace("MessageChunk", "").toLowerCase(),
    } as LangChainMessage;
  }

  const newContent =
    typeof prev.content === "string"
      ? [{ type: "text" as const, text: prev.content }]
      : [...prev.content];

  if (typeof curr?.content === "string") {
    const lastIndex = newContent.length - 1;
    if (newContent[lastIndex]?.type === "text") {
      (newContent[lastIndex] as MessageContentText).text =
        (newContent[lastIndex] as MessageContentText).text + curr.content;
    } else {
      newContent.push({ type: "text", text: curr.content });
    }
  } else if (Array.isArray(curr.content)) {
    const lastIndex = newContent.length - 1;
    for (const item of curr.content) {
      if (!("type" in item)) {
        continue;
      }

      if (item.type === "text") {
        if (newContent[lastIndex]?.type === "text") {
          (newContent[lastIndex] as MessageContentText).text =
            (newContent[lastIndex] as MessageContentText).text + item.text;
        } else {
          newContent.push({ type: "text", text: item.text });
        }
      } else if (item.type === "image_url") {
        newContent.push(item);
      }
    }
  }

  const newToolCalls = [...(prev.tool_calls ?? [])];
  for (const chunk of curr.tool_call_chunks ?? []) {
    const idx = newToolCalls.findIndex(
      (tc) => tc.id != null && tc.id === chunk.id,
    );
    if (idx === -1) {
      const partialJson = chunk.args ?? chunk.args_json ?? "";
      newToolCalls.push({
        ...chunk,
        partial_json: partialJson,
        args: parsePartialJsonObject(partialJson) ?? {},
      });
    } else {
      const existing = newToolCalls[idx]!;
      const partialJson =
        existing.partial_json + (chunk.args ?? chunk.args_json ?? "");
      newToolCalls[idx] = {
        ...chunk,
        ...existing,
        partial_json: partialJson,
        args:
          parsePartialJsonObject(partialJson) ??
          ("args" in existing ? existing.args : {}),
      };
    }
  }

  const newUsageMetadata = mergeUsageMetadata(
    prev.usage_metadata,
    curr.usage_metadata,
  );

  return {
    ...prev,
    content: newContent,
    tool_calls: newToolCalls,
    usage_metadata: newUsageMetadata,
  };
};
