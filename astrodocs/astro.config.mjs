// @ts-check
import { defineConfig } from 'astro/config';
import starlight from '@astrojs/starlight';

// https://astro.build/config
export default defineConfig({
	site: 'https://higgs.local',
	integrations: [
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
