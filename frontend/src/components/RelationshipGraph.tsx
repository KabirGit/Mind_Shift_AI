"use client";

import { useEffect, useMemo, useRef, useState } from "react";
import {
  forceCenter,
  forceCollide,
  forceLink,
  forceManyBody,
  forceSimulation,
  forceX,
  forceY,
  type SimulationLinkDatum,
  type SimulationNodeDatum
} from "d3-force";

import type { PeopleGraph } from "@/lib/api";
import { titleCase } from "@/lib/format";

type GraphNode = PeopleGraph["nodes"][number] &
  SimulationNodeDatum & {
    radius: number;
  };

type GraphLink = PeopleGraph["edges"][number] &
  SimulationLinkDatum<GraphNode>;

const WIDTH = 760;
const HEIGHT = 420;
const CLUSTER_POINTS: Record<string, { x: number; y: number }> = {
  self: { x: WIDTH / 2, y: HEIGHT / 2 },
  family: { x: WIDTH * 0.26, y: HEIGHT * 0.32 },
  friend: { x: WIDTH * 0.72, y: HEIGHT * 0.3 },
  colleague: { x: WIDTH * 0.28, y: HEIGHT * 0.72 },
  partner: { x: WIDTH * 0.72, y: HEIGHT * 0.68 },
  other: { x: WIDTH * 0.5, y: HEIGHT * 0.18 },
  unknown: { x: WIDTH * 0.5, y: HEIGHT * 0.82 }
};

export function RelationshipGraph({ graph }: { graph: PeopleGraph | null }) {
  const [nodes, setNodes] = useState<GraphNode[]>([]);
  const [links, setLinks] = useState<GraphLink[]>([]);
  const tickRef = useRef(0);

  const prepared = useMemo(() => {
    const rawNodes = graph?.nodes ?? [];
    const rawLinks = graph?.edges ?? [];
    const maxMentions = Math.max(1, ...rawNodes.map((node) => node.mention_count));
    const nextNodes: GraphNode[] = rawNodes.map((node) => ({
      ...node,
      radius:
        node.type === "user"
          ? 24
          : 10 + Math.sqrt(node.mention_count / maxMentions) * 18
    }));
    const nextLinks: GraphLink[] = rawLinks.map((edge) => ({ ...edge }));
    return { nodes: nextNodes, links: nextLinks };
  }, [graph]);

  useEffect(() => {
    if (!prepared.nodes.length) {
      return;
    }

    const maxCloseness = Math.max(
      1,
      ...prepared.links.map((link) => link.closeness_score)
    );
    const nodeCopies = prepared.nodes.map((node) => ({ ...node }));
    const linkCopies = prepared.links.map((link) => ({ ...link }));

    const simulation = forceSimulation<GraphNode>(nodeCopies)
      .force(
        "link",
        forceLink<GraphNode, GraphLink>(linkCopies)
          .id((node) => node.id)
          .distance((link) => 185 - (link.closeness_score / maxCloseness) * 105)
          .strength((link) => Math.min(0.8, 0.2 + link.weight * 0.08))
      )
      .force("charge", forceManyBody().strength(-180))
      .force("center", forceCenter(WIDTH / 2, HEIGHT / 2))
      .force(
        "clusterX",
        forceX<GraphNode>((node) => clusterPoint(node).x).strength(0.18)
      )
      .force(
        "clusterY",
        forceY<GraphNode>((node) => clusterPoint(node).y).strength(0.18)
      )
      .force("collide", forceCollide<GraphNode>((node) => node.radius + 10))
      .on("tick", () => {
        tickRef.current += 1;
        if (tickRef.current % 2 === 0) {
          setNodes([...nodeCopies]);
          setLinks([...linkCopies]);
        }
      });

    return () => {
      simulation.stop();
    };
  }, [prepared]);

  const legendTypes = useMemo(() => {
    const types = new Set((graph?.nodes ?? []).map((node) => node.relationship_type));
    return Array.from(types).filter((type) => type !== "self").sort();
  }, [graph]);
  const renderedNodes = prepared.nodes.length ? nodes : [];
  const renderedLinks = prepared.nodes.length ? links : [];

  if (!graph || graph.nodes.length <= 1) {
    return (
      <div className="flex min-h-[320px] items-center justify-center rounded-lg border border-line bg-canvas p-6 text-center text-sm text-muted">
        Add repeated person mentions and the relationship graph will appear here.
      </div>
    );
  }

  return (
    <div className="min-w-0">
      <svg
        aria-label="Relationship graph"
        className="h-[360px] w-full rounded-lg border border-line bg-canvas"
        role="img"
        viewBox={`0 0 ${WIDTH} ${HEIGHT}`}
      >
        <g>
          {renderedLinks.map((link, index) => {
            const source = asNode(link.source);
            const target = asNode(link.target);
            return (
              <line
                key={`${source.id}-${target.id}-${index}`}
                stroke={sentimentColor(link.sentiment)}
                strokeLinecap="round"
                strokeOpacity={0.72}
                strokeWidth={Math.max(2, Math.sqrt(link.weight) * 2.2)}
                x1={source.x ?? WIDTH / 2}
                x2={target.x ?? WIDTH / 2}
                y1={source.y ?? HEIGHT / 2}
                y2={target.y ?? HEIGHT / 2}
              >
                <title>
                  {`${source.label} to ${target.label}: sentiment ${link.sentiment}, closeness ${link.closeness_score}`}
                </title>
              </line>
            );
          })}
        </g>
        <g>
          {renderedNodes.map((node) => (
            <g key={node.id} transform={`translate(${node.x ?? WIDTH / 2},${node.y ?? HEIGHT / 2})`}>
              <circle
                fill={node.type === "user" ? "#1f2933" : relationshipColor(node.relationship_type)}
                r={node.radius}
                stroke="#fffdf8"
                strokeWidth="3"
              >
                <title>
                  {`${node.label}: ${node.mention_count} mentions, ${node.relationship_type}`}
                </title>
              </circle>
              <text
                dy={node.radius + 16}
                fill="#292722"
                fontSize="13"
                fontWeight="700"
                textAnchor="middle"
              >
                {node.label}
              </text>
            </g>
          ))}
        </g>
      </svg>
      <div className="mt-3 flex flex-wrap items-center gap-4 text-xs font-semibold text-muted">
        <span>Sentiment</span>
        <span className="h-2 w-28 rounded-full bg-gradient-to-r from-[#c64545] via-[#e1b94a] to-[#3d915e]" />
        <span>Negative</span>
        <span>Positive</span>
        {legendTypes.map((type) => (
          <span className="inline-flex items-center gap-2" key={type}>
            <span
              className="size-3 rounded-full"
              style={{ backgroundColor: relationshipColor(type) }}
            />
            {titleCase(type)}
          </span>
        ))}
      </div>
    </div>
  );
}

function asNode(value: string | number | GraphNode): GraphNode {
  return value as GraphNode;
}

function clusterPoint(node: GraphNode) {
  return CLUSTER_POINTS[node.relationship_type] ?? CLUSTER_POINTS.unknown;
}

function sentimentColor(sentiment: number): string {
  const clamped = Math.max(-1, Math.min(1, sentiment));
  if (clamped < 0) {
    return interpolate([198, 69, 69], [225, 185, 74], clamped + 1);
  }
  return interpolate([225, 185, 74], [61, 145, 94], clamped);
}

function interpolate(a: number[], b: number[], t: number): string {
  const channels = a.map((start, index) => Math.round(start + (b[index] - start) * t));
  return `rgb(${channels[0]}, ${channels[1]}, ${channels[2]})`;
}

function relationshipColor(type: string): string {
  const colors: Record<string, string> = {
    family: "#5db8a6",
    friend: "#cc785c",
    colleague: "#7c8db5",
    partner: "#b56576",
    other: "#8b8c54",
    unknown: "#a09d96"
  };
  return colors[type] ?? colors.unknown;
}
