// @ts-check
import { defineConfig } from 'astro/config';
import starlight from '@astrojs/starlight';
import mermaid from 'astro-mermaid';

// https://astro.build/config
export default defineConfig({
	site: 'https://higgs.local',
	integrations: [
		// Renders ```mermaid code blocks into real SVG diagrams. Must precede
		// Starlight so it transforms markdown before Starlight processes it.
		mermaid({
			theme: 'dark',
			autoTheme: false,
			mermaidConfig: {
				flowchart: { curve: 'basis', htmlLabels: true, padding: 18, nodeSpacing: 55, rankSpacing: 65 },
				themeVariables: {
					fontFamily: 'ui-sans-serif, system-ui, -apple-system, Segoe UI, sans-serif',
					fontSize: '14px',
					lineColor: '#8b949e',
					clusterBkg: '#161b22',
					clusterBorder: '#30363d',
					titleColor: '#e6edf3',
					textColor: '#e6edf3',
					// sequence-diagram palette (matches the flowchart classDefs)
					actorBkg: '#1f2547',
					actorBorder: '#818cf8',
					actorTextColor: '#c7d2fe',
					signalColor: '#8b949e',
					signalTextColor: '#e6edf3',
					labelBoxBkgColor: '#161b22',
					labelBoxBorderColor: '#30363d',
					noteBkgColor: '#0c2e23',
					noteBorderColor: '#34d399',
					noteTextColor: '#a7f3d0',
				},
			},
		}),
		starlight({
			title: 'higgs — local model runtime',
			social: [],
			sidebar: [
				{
					label: 'Introduction',
					items: [
						{ label: 'What is higgs?', slug: 'overview' },
					],
				},
				{
					label: 'Architecture',
					items: [
						{ label: 'System Design', slug: 'system-design' },
						{ label: 'Remote Fleet', slug: 'remote-fleet' },
						{ label: 'Concurrency model', slug: 'concurrency' },
					],
				},
				{
					label: 'Reference',
					items: [
						{ label: 'Endpoints', slug: 'endpoints' },
					],
				},
				{
					label: 'Guides',
					items: [
						{ label: 'How to Use', slug: 'how-to-use' },
						{ label: 'Development', slug: 'development' },
					],
				},
			],
		}),
	],
});
