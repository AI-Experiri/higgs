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
			theme: 'base',
			autoTheme: true,
			mermaidConfig: {
				flowchart: { curve: 'basis', htmlLabels: true, padding: 18, nodeSpacing: 55, rankSpacing: 70 },
				themeVariables: {
					fontFamily: 'ui-sans-serif, system-ui, -apple-system, Segoe UI, sans-serif',
					fontSize: '14px',
					primaryColor: '#eef2ff',
					primaryBorderColor: '#6366f1',
					primaryTextColor: '#1e1b4b',
					lineColor: '#64748b',
					clusterBkg: '#f8fafc',
					clusterBorder: '#cbd5e1',
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
