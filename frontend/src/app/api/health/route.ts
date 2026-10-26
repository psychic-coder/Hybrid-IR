import { NextResponse } from "next/server";

export const dynamic = "force-dynamic";

/** Liveness probe for the container healthcheck (does not call the backend). */
export function GET() {
  return NextResponse.json({ status: "ok" });
}
