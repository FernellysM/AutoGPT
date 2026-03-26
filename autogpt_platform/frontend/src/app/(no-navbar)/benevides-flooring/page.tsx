"use client";

import { useEffect, useRef, useState } from "react";

// ─── Animated counter hook ────────────────────────────────────────────────────
function useCountUp(target: number, duration = 2000, start = false) {
  const [count, setCount] = useState(0);
  useEffect(() => {
    if (!start) return;
    let startTime: number | null = null;
    const step = (ts: number) => {
      if (!startTime) startTime = ts;
      const progress = Math.min((ts - startTime) / duration, 1);
      setCount(Math.floor(progress * target));
      if (progress < 1) requestAnimationFrame(step);
    };
    requestAnimationFrame(step);
  }, [target, duration, start]);
  return count;
}

// ─── Intersection observer hook ───────────────────────────────────────────────
function useInView(threshold = 0.2) {
  const ref = useRef<HTMLDivElement>(null);
  const [inView, setInView] = useState(false);
  useEffect(() => {
    const el = ref.current;
    if (!el) return;
    const obs = new IntersectionObserver(
      ([entry]) => {
        if (entry.isIntersecting) {
          setInView(true);
          obs.disconnect();
        }
      },
      { threshold },
    );
    obs.observe(el);
    return () => obs.disconnect();
  }, [threshold]);
  return { ref, inView };
}

// ─── SVG Illustrations ────────────────────────────────────────────────────────

function HexTilePattern() {
  return (
    <svg
      viewBox="0 0 400 340"
      fill="none"
      xmlns="http://www.w3.org/2000/svg"
      className="h-full w-full"
    >
      <defs>
        <linearGradient id="tileGrad1" x1="0" y1="0" x2="1" y2="1">
          <stop offset="0%" stopColor="#C8A97E" />
          <stop offset="100%" stopColor="#8B6C42" />
        </linearGradient>
        <linearGradient id="tileGrad2" x1="0" y1="0" x2="1" y2="1">
          <stop offset="0%" stopColor="#E8D5B7" />
          <stop offset="100%" stopColor="#C8A97E" />
        </linearGradient>
        <linearGradient id="tileGrad3" x1="1" y1="0" x2="0" y2="1">
          <stop offset="0%" stopColor="#6B4F2A" />
          <stop offset="100%" stopColor="#8B6C42" />
        </linearGradient>
      </defs>
      {/* Hexagonal tile grid */}
      {[
        [60, 55],
        [130, 55],
        [200, 55],
        [270, 55],
        [340, 55],
        [25, 110],
        [95, 110],
        [165, 110],
        [235, 110],
        [305, 110],
        [375, 110],
        [60, 165],
        [130, 165],
        [200, 165],
        [270, 165],
        [340, 165],
        [25, 220],
        [95, 220],
        [165, 220],
        [235, 220],
        [305, 220],
        [375, 220],
        [60, 275],
        [130, 275],
        [200, 275],
        [270, 275],
        [340, 275],
      ].map(([cx, cy], i) => {
        const grads = ["tileGrad1", "tileGrad2", "tileGrad3"];
        return (
          <polygon
            key={i}
            points={`${cx},${cy - 30} ${cx + 26},${cy - 15} ${cx + 26},${cy + 15} ${cx},${cy + 30} ${cx - 26},${cy + 15} ${cx - 26},${cy - 15}`}
            fill={`url(#${grads[i % 3]})`}
            stroke="#fff"
            strokeWidth="1.5"
            opacity={0.85}
          />
        );
      })}
    </svg>
  );
}

function WoodPlankPattern() {
  return (
    <svg
      viewBox="0 0 480 320"
      fill="none"
      xmlns="http://www.w3.org/2000/svg"
      className="h-full w-full"
    >
      <defs>
        <linearGradient id="plank1" x1="0" y1="0" x2="0" y2="1">
          <stop offset="0%" stopColor="#D4A76A" />
          <stop offset="50%" stopColor="#B8843A" />
          <stop offset="100%" stopColor="#8B6320" />
        </linearGradient>
        <linearGradient id="plank2" x1="0" y1="0" x2="0" y2="1">
          <stop offset="0%" stopColor="#C8945A" />
          <stop offset="50%" stopColor="#A0702E" />
          <stop offset="100%" stopColor="#7A5018" />
        </linearGradient>
        <linearGradient id="plank3" x1="0" y1="0" x2="0" y2="1">
          <stop offset="0%" stopColor="#E0B880" />
          <stop offset="50%" stopColor="#C49040" />
          <stop offset="100%" stopColor="#9A7028" />
        </linearGradient>
      </defs>
      {/* Wood planks with staggered offset */}
      {[0, 1, 2, 3, 4, 5, 6, 7].map((row) => {
        const y = row * 42;
        const offset = row % 2 === 0 ? 0 : -120;
        const grads = ["plank1", "plank2", "plank3"];
        return (
          <g key={row}>
            {[-1, 0, 1, 2, 3].map((col) => {
              const x = col * 240 + offset;
              return (
                <g key={col}>
                  <rect
                    x={x}
                    y={y}
                    width={235}
                    height={40}
                    fill={`url(#${grads[(row + col) % 3]})`}
                    rx={2}
                  />
                  {/* Wood grain lines */}
                  <line
                    x1={x + 20}
                    y1={y + 8}
                    x2={x + 200}
                    y2={y + 6}
                    stroke="rgba(0,0,0,0.08)"
                    strokeWidth="1"
                  />
                  <line
                    x1={x + 30}
                    y1={y + 18}
                    x2={x + 190}
                    y2={y + 22}
                    stroke="rgba(0,0,0,0.06)"
                    strokeWidth="1"
                  />
                  <line
                    x1={x + 15}
                    y1={y + 30}
                    x2={x + 210}
                    y2={y + 32}
                    stroke="rgba(0,0,0,0.07)"
                    strokeWidth="1"
                  />
                  {/* Plank gap */}
                  <rect
                    x={x + 235}
                    y={y}
                    width={5}
                    height={40}
                    fill="rgba(0,0,0,0.15)"
                  />
                </g>
              );
            })}
          </g>
        );
      })}
    </svg>
  );
}

function MarblePattern() {
  return (
    <svg
      viewBox="0 0 400 400"
      fill="none"
      xmlns="http://www.w3.org/2000/svg"
      className="h-full w-full"
    >
      <defs>
        <radialGradient id="marbleBase" cx="50%" cy="50%" r="70%">
          <stop offset="0%" stopColor="#F5F0EA" />
          <stop offset="60%" stopColor="#E8E0D5" />
          <stop offset="100%" stopColor="#D4C8B8" />
        </radialGradient>
        <linearGradient id="vein1" x1="0" y1="0" x2="1" y2="1">
          <stop offset="0%" stopColor="#C8B89A" stopOpacity="0" />
          <stop offset="40%" stopColor="#A89070" stopOpacity="0.5" />
          <stop offset="100%" stopColor="#C8B89A" stopOpacity="0" />
        </linearGradient>
      </defs>
      <rect width="400" height="400" fill="url(#marbleBase)" />
      {/* Marble veins */}
      <path
        d="M-10 80 Q60 90 120 70 Q180 50 240 85 Q300 120 380 100 Q420 90 440 95"
        stroke="url(#vein1)"
        strokeWidth="3"
        fill="none"
        opacity="0.6"
      />
      <path
        d="M-10 180 Q80 160 150 200 Q220 240 290 190 Q350 150 420 170"
        stroke="#B0A090"
        strokeWidth="2"
        fill="none"
        opacity="0.4"
      />
      <path
        d="M50 -10 Q70 80 60 160 Q50 240 80 340 Q100 400 90 440"
        stroke="#C0B0A0"
        strokeWidth="1.5"
        fill="none"
        opacity="0.35"
      />
      <path
        d="M200 -10 Q180 100 220 200 Q260 300 240 400"
        stroke="#A89878"
        strokeWidth="2.5"
        fill="none"
        opacity="0.3"
      />
      <path
        d="M350 20 Q320 120 360 200 Q400 280 370 400"
        stroke="#C0B0A0"
        strokeWidth="1.5"
        fill="none"
        opacity="0.4"
      />
      {/* Subtle texture dots */}
      {Array.from({ length: 30 }, (_, i) => (
        <circle
          key={i}
          cx={(i * 137) % 400}
          cy={(i * 97 + 50) % 400}
          r={Math.random() * 2 + 0.5}
          fill="#A89878"
          opacity={0.15}
        />
      ))}
    </svg>
  );
}

function LogoMark() {
  return (
    <svg
      viewBox="0 0 80 80"
      fill="none"
      xmlns="http://www.w3.org/2000/svg"
      className="h-full w-full"
    >
      <defs>
        <linearGradient id="logoGrad" x1="0" y1="0" x2="1" y2="1">
          <stop offset="0%" stopColor="#C8A97E" />
          <stop offset="100%" stopColor="#6B4F2A" />
        </linearGradient>
      </defs>
      {/* Geometric B mark made of floor tiles */}
      <rect x="10" y="10" width="25" height="25" rx="3" fill="url(#logoGrad)" />
      <rect
        x="10"
        y="45"
        width="25"
        height="25"
        rx="3"
        fill="url(#logoGrad)"
        opacity="0.85"
      />
      <rect
        x="45"
        y="10"
        width="25"
        height="25"
        rx="3"
        fill="url(#logoGrad)"
        opacity="0.7"
      />
      <rect
        x="45"
        y="45"
        width="25"
        height="25"
        rx="3"
        fill="url(#logoGrad)"
        opacity="0.55"
      />
      <rect
        x="27"
        y="27"
        width="26"
        height="26"
        rx="3"
        fill="#fff"
        opacity="0.12"
      />
    </svg>
  );
}

// ─── Stats component ──────────────────────────────────────────────────────────
function StatCounter({
  value,
  suffix,
  label,
  inView,
}: {
  value: number;
  suffix: string;
  label: string;
  inView: boolean;
}) {
  const count = useCountUp(value, 2200, inView);
  return (
    <div className="flex flex-col items-center gap-2">
      <span className="text-5xl font-semibold tracking-tight text-white">
        {count}
        {suffix}
      </span>
      <span className="text-sm uppercase tracking-widest text-white/50">
        {label}
      </span>
    </div>
  );
}

// ─── Main page ────────────────────────────────────────────────────────────────
export default function BenevidesFlooringPage() {
  const [scrollY, setScrollY] = useState(0);
  const [heroVisible, setHeroVisible] = useState(false);

  const heroRef = useRef<HTMLDivElement>(null);
  const section2 = useInView(0.15);
  const section3 = useInView(0.15);
  const section4 = useInView(0.15);
  const statsSection = useInView(0.3);

  useEffect(() => {
    const timer = setTimeout(() => setHeroVisible(true), 100);
    return () => clearTimeout(timer);
  }, []);

  useEffect(() => {
    const onScroll = () => setScrollY(window.scrollY);
    window.addEventListener("scroll", onScroll, { passive: true });
    return () => window.removeEventListener("scroll", onScroll);
  }, []);

  const heroParallax = scrollY * 0.4;

  return (
    <main className="overflow-x-hidden bg-[#0a0a0a] text-white">
      {/* ── NAVIGATION ──────────────────────────────────────────────────────── */}
      <nav
        className="fixed left-0 right-0 top-0 z-50 flex items-center justify-between px-8 py-5 transition-all duration-500"
        style={{
          background:
            scrollY > 60
              ? "rgba(10,10,10,0.85)"
              : "linear-gradient(to bottom, rgba(0,0,0,0.5), transparent)",
          backdropFilter: scrollY > 60 ? "blur(16px)" : "none",
        }}
      >
        <div className="flex items-center gap-3">
          <div className="h-9 w-9">
            <LogoMark />
          </div>
          <span className="text-sm font-medium uppercase tracking-[0.2em] text-white/90">
            Benevides Flooring
          </span>
        </div>
        <div className="hidden items-center gap-8 text-xs uppercase tracking-widest text-white/60 md:flex">
          <a href="#collections" className="transition-colors hover:text-white">
            Collections
          </a>
          <a href="#process" className="transition-colors hover:text-white">
            Process
          </a>
          <a href="#gallery" className="transition-colors hover:text-white">
            Gallery
          </a>
          <a href="#contact" className="transition-colors hover:text-white">
            Contact
          </a>
        </div>
        <button className="rounded-full border border-white/30 px-5 py-2.5 text-xs uppercase tracking-widest transition-all duration-300 hover:border-white/80 hover:bg-white hover:text-black">
          Get a Quote
        </button>
      </nav>

      {/* ── HERO ────────────────────────────────────────────────────────────── */}
      <section
        ref={heroRef}
        className="relative flex h-screen flex-col items-center justify-center overflow-hidden"
      >
        {/* Parallax background illustration */}
        <div
          className="absolute inset-0 opacity-20"
          style={{ transform: `translateY(${heroParallax}px)` }}
        >
          <WoodPlankPattern />
        </div>

        {/* Radial glow */}
        <div className="absolute inset-0 bg-[radial-gradient(ellipse_60%_60%_at_50%_50%,rgba(200,169,126,0.15),transparent)]" />
        <div className="absolute bottom-0 left-0 right-0 h-40 bg-gradient-to-t from-[#0a0a0a] to-transparent" />

        {/* Content */}
        <div className="relative z-10 flex max-w-5xl flex-col items-center px-6 text-center">
          {/* Logo mark */}
          <div
            className="mb-10 h-20 w-20 transition-all duration-1000"
            style={{
              opacity: heroVisible ? 1 : 0,
              transform: heroVisible ? "scale(1)" : "scale(0.8)",
            }}
          >
            <LogoMark />
          </div>

          {/* Eyebrow */}
          <p
            className="mb-6 text-xs uppercase tracking-[0.4em] text-[#C8A97E] transition-all delay-200 duration-700"
            style={{
              opacity: heroVisible ? 1 : 0,
              transform: heroVisible ? "translateY(0)" : "translateY(20px)",
            }}
          >
            Luxury Flooring Since 1998
          </p>

          {/* Headline */}
          <h1
            className="mb-8 text-6xl font-semibold leading-[0.95] tracking-[-0.03em] transition-all delay-300 duration-1000 md:text-8xl lg:text-[108px]"
            style={{
              opacity: heroVisible ? 1 : 0,
              transform: heroVisible ? "translateY(0)" : "translateY(40px)",
            }}
          >
            <span className="block">Where Every</span>
            <span
              className="block"
              style={{
                background:
                  "linear-gradient(135deg, #E8D5B7 0%, #C8A97E 40%, #8B6C42 100%)",
                WebkitBackgroundClip: "text",
                WebkitTextFillColor: "transparent",
                backgroundClip: "text",
              }}
            >
              Step Tells
            </span>
            <span className="block">Your Story.</span>
          </h1>

          {/* Subheadline */}
          <p
            className="mb-12 max-w-xl text-lg leading-relaxed text-white/50 transition-all delay-500 duration-1000 md:text-xl"
            style={{
              opacity: heroVisible ? 1 : 0,
              transform: heroVisible ? "translateY(0)" : "translateY(30px)",
            }}
          >
            Handcrafted floors that transform spaces into experiences. Premium
            hardwood, stone, and tile — installed with uncompromising precision.
          </p>

          {/* CTA buttons */}
          <div
            className="flex flex-col gap-4 transition-all delay-700 duration-1000 sm:flex-row"
            style={{
              opacity: heroVisible ? 1 : 0,
              transform: heroVisible ? "translateY(0)" : "translateY(20px)",
            }}
          >
            <button className="rounded-full bg-[#C8A97E] px-8 py-4 text-sm font-medium tracking-wide text-black transition-all duration-300 hover:scale-105 hover:bg-[#E8C99E]">
              Explore Collections
            </button>
            <button className="rounded-full border border-white/30 px-8 py-4 text-sm font-medium tracking-wide transition-all duration-300 hover:border-white/70 hover:bg-white/5">
              Watch Our Story
            </button>
          </div>
        </div>

        {/* Scroll indicator */}
        <div
          className="absolute bottom-10 flex flex-col items-center gap-2 transition-all delay-1000 duration-1000"
          style={{ opacity: heroVisible ? 0.4 : 0 }}
        >
          <span className="text-xs uppercase tracking-widest">Scroll</span>
          <div className="h-12 w-px animate-pulse bg-gradient-to-b from-white to-transparent" />
        </div>
      </section>

      {/* ── SECTION 2 — INTRO STATEMENT ─────────────────────────────────────── */}
      <section
        id="collections"
        className="relative flex flex-col items-center px-6 py-40 text-center"
      >
        <div
          ref={section2.ref}
          className="max-w-4xl transition-all duration-1000"
          style={{
            opacity: section2.inView ? 1 : 0,
            transform: section2.inView ? "translateY(0)" : "translateY(60px)",
          }}
        >
          <p className="mb-8 text-xs uppercase tracking-[0.4em] text-[#C8A97E]">
            The Benevides Difference
          </p>
          <h2 className="mb-10 text-4xl font-semibold leading-[1.05] tracking-[-0.02em] md:text-6xl lg:text-7xl">
            Floors designed to{" "}
            <span
              style={{
                background:
                  "linear-gradient(135deg, #E8D5B7 0%, #C8A97E 60%, #8B6C42 100%)",
                WebkitBackgroundClip: "text",
                WebkitTextFillColor: "transparent",
                backgroundClip: "text",
              }}
            >
              last a lifetime.
            </span>
          </h2>
          <p className="mx-auto max-w-2xl text-lg leading-relaxed text-white/50">
            We source only the finest materials from around the world — rich
            Brazilian hardwoods, Italian marble, hand-glazed ceramic — then
            craft each installation with the detail and care your home deserves.
          </p>
        </div>
      </section>

      {/* ── SECTION 3 — COLLECTIONS GRID ────────────────────────────────────── */}
      <section className="mx-auto max-w-7xl px-6 py-20" id="gallery">
        <div
          ref={section3.ref}
          className="grid grid-cols-1 gap-6 transition-all duration-1000 md:grid-cols-3"
          style={{
            opacity: section3.inView ? 1 : 0,
            transform: section3.inView ? "translateY(0)" : "translateY(60px)",
          }}
        >
          {/* Card 1 — Hardwood */}
          <div className="group relative aspect-[3/4] cursor-pointer overflow-hidden rounded-3xl bg-[#141414]">
            <div className="absolute inset-0 transition-transform duration-700 group-hover:scale-105">
              <WoodPlankPattern />
            </div>
            <div className="absolute inset-0 bg-gradient-to-t from-black/90 via-black/20 to-transparent" />
            <div className="absolute bottom-0 left-0 right-0 p-8">
              <span className="mb-2 block text-xs uppercase tracking-widest text-[#C8A97E]">
                Collection 01
              </span>
              <h3 className="mb-2 text-2xl font-semibold">Artisan Hardwood</h3>
              <p className="mb-5 text-sm text-white/50">
                Hand-scraped, wire-brushed, and UV-cured. Brazilian cherry to
                white oak.
              </p>
              <span className="text-xs uppercase tracking-widest text-white/40 transition-colors group-hover:text-[#C8A97E]">
                Discover →
              </span>
            </div>
          </div>

          {/* Card 2 — Stone & Marble */}
          <div className="group relative aspect-[3/4] cursor-pointer overflow-hidden rounded-3xl bg-[#141414] md:mt-12">
            <div className="absolute inset-0 transition-transform duration-700 group-hover:scale-105">
              <MarblePattern />
            </div>
            <div className="absolute inset-0 bg-gradient-to-t from-black/90 via-black/20 to-transparent" />
            <div className="absolute bottom-0 left-0 right-0 p-8">
              <span className="mb-2 block text-xs uppercase tracking-widest text-[#C8A97E]">
                Collection 02
              </span>
              <h3 className="mb-2 text-2xl font-semibold">Stone & Marble</h3>
              <p className="mb-5 text-sm text-white/50">
                Calacatta, travertine, slate. Natural surfaces refined to
                perfection.
              </p>
              <span className="text-xs uppercase tracking-widest text-white/40 transition-colors group-hover:text-[#C8A97E]">
                Discover →
              </span>
            </div>
          </div>

          {/* Card 3 — Artisan Tile */}
          <div className="group relative aspect-[3/4] cursor-pointer overflow-hidden rounded-3xl bg-[#141414]">
            <div className="absolute inset-0 transition-transform duration-700 group-hover:scale-105">
              <HexTilePattern />
            </div>
            <div className="absolute inset-0 bg-gradient-to-t from-black/90 via-black/20 to-transparent" />
            <div className="absolute bottom-0 left-0 right-0 p-8">
              <span className="mb-2 block text-xs uppercase tracking-widest text-[#C8A97E]">
                Collection 03
              </span>
              <h3 className="mb-2 text-2xl font-semibold">Artisan Tile</h3>
              <p className="mb-5 text-sm text-white/50">
                Geometric patterns, handmade ceramics, and encaustic cement
                tiles.
              </p>
              <span className="text-xs uppercase tracking-widest text-white/40 transition-colors group-hover:text-[#C8A97E]">
                Discover →
              </span>
            </div>
          </div>
        </div>
      </section>

      {/* ── SECTION 4 — PROCESS ─────────────────────────────────────────────── */}
      <section id="process" className="px-6 py-40">
        <div
          ref={section4.ref}
          className="mx-auto max-w-6xl"
          style={{
            opacity: section4.inView ? 1 : 0,
            transform: section4.inView ? "translateY(0)" : "translateY(60px)",
            transition: "all 1s ease",
          }}
        >
          <p className="mb-6 text-center text-xs uppercase tracking-[0.4em] text-[#C8A97E]">
            Our Process
          </p>
          <h2 className="mb-20 text-center text-4xl font-semibold tracking-[-0.02em] md:text-6xl">
            From vision to reality.
          </h2>

          <div className="grid grid-cols-1 gap-8 md:grid-cols-4">
            {[
              {
                num: "01",
                title: "Consultation",
                desc: "We visit your space, understand your style, and recommend materials that fit your life.",
              },
              {
                num: "02",
                title: "Design",
                desc: "Our designers create a floor plan with samples, layouts, and 3D visualizations.",
              },
              {
                num: "03",
                title: "Sourcing",
                desc: "We procure the finest materials directly from mills and quarries worldwide.",
              },
              {
                num: "04",
                title: "Installation",
                desc: "Certified craftsmen install with precision — clean, quiet, and on schedule.",
              },
            ].map((step, i) => (
              <div
                key={i}
                className="relative"
                style={{
                  transitionDelay: `${i * 150}ms`,
                  opacity: section4.inView ? 1 : 0,
                  transform: section4.inView
                    ? "translateY(0)"
                    : "translateY(40px)",
                  transition: "all 0.8s ease",
                }}
              >
                <div className="mb-4 text-6xl font-semibold text-white/5">
                  {step.num}
                </div>
                <div className="mb-5 h-px w-8 bg-[#C8A97E]" />
                <h3 className="mb-3 text-xl font-medium">{step.title}</h3>
                <p className="text-sm leading-relaxed text-white/40">
                  {step.desc}
                </p>
              </div>
            ))}
          </div>
        </div>
      </section>

      {/* ── STATS SECTION ───────────────────────────────────────────────────── */}
      <section className="relative overflow-hidden px-6 py-32">
        <div className="absolute inset-0 bg-[#C8A97E]/5" />
        <div className="absolute inset-0 bg-[radial-gradient(ellipse_80%_50%_at_50%_50%,rgba(200,169,126,0.08),transparent)]" />

        <div
          ref={statsSection.ref}
          className="relative mx-auto grid max-w-5xl grid-cols-2 gap-16 text-center md:grid-cols-4"
        >
          <StatCounter
            value={25}
            suffix="+"
            label="Years of Excellence"
            inView={statsSection.inView}
          />
          <StatCounter
            value={4800}
            suffix="+"
            label="Projects Completed"
            inView={statsSection.inView}
          />
          <StatCounter
            value={98}
            suffix="%"
            label="Client Satisfaction"
            inView={statsSection.inView}
          />
          <StatCounter
            value={12}
            suffix=""
            label="Design Awards"
            inView={statsSection.inView}
          />
        </div>
      </section>

      {/* ── TESTIMONIAL ─────────────────────────────────────────────────────── */}
      <section className="flex flex-col items-center px-6 py-40 text-center">
        <div className="max-w-3xl">
          <div className="mb-10 flex justify-center gap-1">
            {Array.from({ length: 5 }, (_, i) => (
              <svg
                key={i}
                viewBox="0 0 20 20"
                fill="#C8A97E"
                className="h-5 w-5"
              >
                <path d="M9.049 2.927c.3-.921 1.603-.921 1.902 0l1.07 3.292a1 1 0 00.95.69h3.462c.969 0 1.371 1.24.588 1.81l-2.8 2.034a1 1 0 00-.364 1.118l1.07 3.292c.3.921-.755 1.688-1.54 1.118l-2.8-2.034a1 1 0 00-1.175 0l-2.8 2.034c-.784.57-1.838-.197-1.539-1.118l1.07-3.292a1 1 0 00-.364-1.118L2.98 8.72c-.783-.57-.38-1.81.588-1.81h3.461a1 1 0 00.951-.69l1.07-3.292z" />
              </svg>
            ))}
          </div>
          <blockquote className="mb-10 text-3xl font-light leading-relaxed tracking-[-0.01em] text-white/90 md:text-4xl">
            &ldquo;Benevides didn&apos;t just install floors — they elevated our
            entire home. The craftsmanship is extraordinary, and the team was a
            pleasure to work with from start to finish.&rdquo;
          </blockquote>
          <div className="flex flex-col items-center gap-1">
            <span className="text-sm font-medium text-white/80">
              Alexandra & Marcus Rivera
            </span>
            <span className="text-xs uppercase tracking-widest text-white/30">
              Residential Client, Miami FL
            </span>
          </div>
        </div>
      </section>

      {/* ── CTA FINAL ───────────────────────────────────────────────────────── */}
      <section
        id="contact"
        className="relative flex flex-col items-center overflow-hidden px-6 py-40 text-center"
      >
        {/* Background tile illustration */}
        <div className="absolute inset-0 opacity-10">
          <HexTilePattern />
        </div>
        <div className="absolute inset-0 bg-gradient-to-b from-[#0a0a0a] via-transparent to-[#0a0a0a]" />

        <div className="relative z-10 max-w-3xl">
          <p className="mb-6 text-xs uppercase tracking-[0.4em] text-[#C8A97E]">
            Begin Your Project
          </p>
          <h2 className="mb-8 text-5xl font-semibold leading-[1.05] tracking-[-0.02em] md:text-7xl">
            Ready to transform
            <br />
            your space?
          </h2>
          <p className="mx-auto mb-12 max-w-xl text-lg leading-relaxed text-white/40">
            Schedule a free in-home consultation with one of our design
            specialists. We&apos;ll bring samples, inspiration, and ideas
            tailored specifically to you.
          </p>
          <div className="flex flex-col justify-center gap-4 sm:flex-row">
            <button className="rounded-full bg-[#C8A97E] px-10 py-4 text-sm font-medium tracking-wide text-black transition-all duration-300 hover:scale-105 hover:bg-[#E8C99E]">
              Book a Free Consultation
            </button>
            <button className="rounded-full border border-white/20 px-10 py-4 text-sm font-medium tracking-wide transition-all duration-300 hover:border-white/50 hover:bg-white/5">
              Call (305) 555-0198
            </button>
          </div>
        </div>
      </section>

      {/* ── FOOTER ──────────────────────────────────────────────────────────── */}
      <footer className="flex flex-col items-center justify-between gap-6 border-t border-white/5 px-8 py-12 text-xs text-white/30 md:flex-row">
        <div className="flex items-center gap-3">
          <div className="h-7 w-7">
            <LogoMark />
          </div>
          <span className="uppercase tracking-widest">Benevides Flooring</span>
        </div>
        <nav className="flex gap-8 uppercase tracking-widest">
          <a href="#" className="transition-colors hover:text-white/70">
            Privacy
          </a>
          <a href="#" className="transition-colors hover:text-white/70">
            Terms
          </a>
          <a href="#" className="transition-colors hover:text-white/70">
            Instagram
          </a>
          <a href="#" className="transition-colors hover:text-white/70">
            Houzz
          </a>
        </nav>
        <span>© 2025 Benevides Flooring. All rights reserved.</span>
      </footer>

      {/* ── GLOBAL STYLES ───────────────────────────────────────────────────── */}
      <style>{`
        @keyframes float {
          0%, 100% { transform: translateY(0px); }
          50% { transform: translateY(-12px); }
        }
        html {
          scroll-behavior: smooth;
        }
      `}</style>
    </main>
  );
}
