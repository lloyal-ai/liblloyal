# Publishing API Documentation

The API documentation is automatically published to GitHub Pages on every push to `main`.

## Automatic Publishing (Recommended)

**Setup (one-time):**

1. Go to your GitHub repository settings
2. Navigate to **Settings** → **Pages**
3. Under "Build and deployment":
   - Source: **GitHub Actions**
4. Save changes

**That's it!** Now on every push to `main` that modifies:
- `include/lloyal/**` (any header)
- `Doxyfile` (Doxygen config)
- `.github/workflows/docs.yml` (workflow itself)

The workflow will:
1. Generate fresh documentation with Doxygen
2. Deploy to GitHub Pages
3. Your docs are live at: `https://<username>.github.io/<repo>/`

**Manual trigger:**
You can also manually trigger the workflow from GitHub:
- Go to **Actions** tab
- Select "Generate and Deploy API Docs"
- Click "Run workflow"

## Local Preview Before Publishing

```bash
# Generate locally to preview
./scripts/generate-docs.sh
open docs/api/html/index.html

# Commit header changes when satisfied
git add include/lloyal/your_file.hpp
git commit -m "docs: update API documentation"
git push origin main

# GitHub Actions will auto-generate and deploy
```

## Workflow Details

The `.github/workflows/docs.yml` workflow:

1. **Triggers on:**
   - Push to `main` (when headers or config change)
   - Manual dispatch

2. **Build job:**
   - Checks out code
   - Installs Doxygen + Graphviz
   - Runs `doxygen Doxyfile`
   - Uploads generated HTML as artifact

3. **Deploy job:**
   - Takes HTML artifact
   - Deploys to GitHub Pages
   - Updates live site

## Custom Domain (Optional)

To use a custom domain like `docs.yourproject.com`:

1. Add a `docs/api/html/CNAME` file:
   ```
   docs.yourproject.com
   ```

2. Configure DNS:
   ```
   CNAME docs.yourproject.com -> <username>.github.io
   ```

3. Enable in GitHub Settings → Pages → Custom domain

## Troubleshooting

**"Pages build and deployment failed"**
- Check Actions tab for error details
- Verify GitHub Pages is enabled (Settings → Pages)
- Ensure Pages source is set to "GitHub Actions"

**"Documentation not updating"**
- Workflow only triggers on `main` branch
- Only runs when headers or config change
- Check Actions tab to see if workflow ran
- Try manual trigger via Actions tab

**"404 Not Found"**
- Wait 1-2 minutes after first deployment
- URL is `https://<username>.github.io/<repo>/`
- Check repository settings for correct Pages URL

**"Doxygen warnings"**
- These don't prevent deployment
- Check Actions logs for details
- Fix warnings in header comments
- Push to regenerate

## Monitoring

- **Actions tab** - See build/deploy history
- **Deployments** - See active deployments and URLs
- **Pages settings** - See current configuration

## Disabling Auto-Publish

To disable automatic publishing:

1. Delete or disable `.github/workflows/docs.yml`
2. Or change the `on:` triggers to only `workflow_dispatch`

## Alternative: Manual Deployment

If you prefer manual control:

```bash
# Generate docs
./scripts/generate-docs.sh

# Create gh-pages branch (first time only)
git checkout --orphan gh-pages
git rm -rf .
cp -r docs/api/html/* .
git add .
git commit -m "docs: initial API documentation"
git push origin gh-pages

# Update docs (later)
git checkout main
./scripts/generate-docs.sh
git checkout gh-pages
rm -rf *
cp -r docs/api/html/* .
git add .
git commit -m "docs: update API documentation"
git push origin gh-pages
```

Then set GitHub Pages source to "gh-pages branch" in settings.

**Recommendation:** Use automatic GitHub Actions - it's simpler and always up-to-date.
